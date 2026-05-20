import { DurableObject } from 'cloudflare:workers';
import { isAdminAuthenticated } from './auth';

class HttpError extends Error {
	status: number;
	constructor(message: string, status: number) {
		super(message);
		this.name = this.constructor.name;
		this.status = status;
	}
}

const fixCors = ({ headers, status, statusText }: { headers?: HeadersInit; status?: number; statusText?: string }) => {
	const newHeaders = new Headers(headers);
	newHeaders.set('Access-Control-Allow-Origin', '*');
	newHeaders.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
	newHeaders.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-goog-api-key');
	return { headers: newHeaders, status, statusText };
};

const BASE_URL = 'https://generativelanguage.googleapis.com';
const API_VERSION = 'v1beta';
// Google 官方 OpenAI 兼容端点
const OPENAI_COMPAT_BASE = `${BASE_URL}/${API_VERSION}/openai`;

const isForwardClientKeyEnabled = (value: unknown): boolean => {
	if (typeof value === 'boolean') return value;
	if (typeof value !== 'string') return false;
	return ['true', '1', 'yes', 'on'].includes(value.trim().toLowerCase());
};

const makeHeaders = (apiKey: string, more?: Record<string, string>) => ({
	'x-goog-api-client': 'genai-js/0.21.0',
	...(apiKey && { 'x-goog-api-key': apiKey }),
	...more,
});

// =================================================================================================
// Google OpenAI 兼容端点响应补丁
//
// 问题1: tool_calls 缺少 index 字段（Vercel AI SDK 要求必须有）
// 问题2: tool_calls 带有 extra_content.google.thought_signature，
//        后续多轮对话中必须回传，否则 Google 返回 400 错误。
//        但 Vercel AI SDK 不认识这个字段，会在序列化时丢弃。
//
// 解决方案:
// - 响应端: 从 tool_calls 中提取 thought_signature 并缓存，然后移除 extra_content
//           （SDK 不认识会报错），同时补上缺失的 index
// - 请求端: 在发送给 Google 前，将缓存的 thought_signature 注入回 assistant 消息的 tool_calls
// =================================================================================================

type SignatureCache = Map<string, { signature: string; savedAt: number }>;
const SIGNATURE_TTL = 6 * 60 * 60 * 1000; // 6 小时

/**
 * 从响应中提取 thought_signature 并缓存，同时清理 SDK 不认识的字段、补 index
 */
function patchResponseBody(body: any, signatureCache: SignatureCache) {
	if (!body?.choices) return;
	for (const choice of body.choices) {
		patchToolCallsInPlace(choice?.message?.tool_calls, signatureCache);
		patchToolCallsInPlace(choice?.delta?.tool_calls, signatureCache);
	}
}

function patchToolCallsInPlace(toolCalls: any[] | undefined, signatureCache: SignatureCache) {
	if (!Array.isArray(toolCalls)) return;
	for (let i = 0; i < toolCalls.length; i++) {
		const tc = toolCalls[i];
		if (!tc) continue;

		// 补 index
		if (tc.index === undefined) {
			tc.index = i;
		}

		// 提取并缓存 thought_signature
		const sig = tc.extra_content?.google?.thought_signature;
		if (sig && tc.id) {
			signatureCache.set(tc.id, { signature: sig, savedAt: Date.now() });
			console.log(`[patchToolCalls] Cached thought_signature for tool_call ${tc.id}`);
		}

		// 移除 extra_content（SDK 不认识，留着会导致问题）
		if (tc.extra_content) {
			delete tc.extra_content;
		}
	}
}

/**
 * 在发送给 Google 前，将缓存的 thought_signature 注入回请求体
 */
function injectSignaturesIntoRequest(body: any, signatureCache: SignatureCache) {
	if (!body?.messages || !Array.isArray(body.messages)) return;
	let injected = 0;

	for (const msg of body.messages) {
		if (msg.role !== 'assistant' || !Array.isArray(msg.tool_calls)) continue;

		for (const tc of msg.tool_calls) {
			if (!tc.id) continue;
			const cached = signatureCache.get(tc.id);
			if (cached && Date.now() - cached.savedAt < SIGNATURE_TTL) {
				// 注入 extra_content.google.thought_signature
				tc.extra_content = tc.extra_content || {};
				tc.extra_content.google = tc.extra_content.google || {};
				tc.extra_content.google.thought_signature = cached.signature;
				injected++;
			}
		}
	}

	if (injected > 0) {
		console.log(`[injectSignatures] Injected ${injected} thought_signature(s) into request`);
	}
}

/**
 * 清理过期的缓存条目
 */
function cleanupSignatureCache(cache: SignatureCache) {
	const now = Date.now();
	for (const [key, val] of cache) {
		if (now - val.savedAt > SIGNATURE_TTL) {
			cache.delete(key);
		}
	}
}

/**
 * 创建带 buffer 的 SSE 补丁 TransformStream
 */
function createPatchTransform(signatureCache: SignatureCache): Transformer<string, string> {
	let buffer = '';
	return {
		transform(chunk: string, controller: TransformStreamDefaultController<string>) {
			buffer += chunk;
			const lines = buffer.split('\n');
			buffer = lines.pop()!;

			for (const line of lines) {
				controller.enqueue(patchSSELine(line, signatureCache) + '\n');
			}
		},
		flush(controller: TransformStreamDefaultController<string>) {
			if (buffer) {
				controller.enqueue(patchSSELine(buffer, signatureCache));
			}
		},
	};
}

function patchSSELine(line: string, signatureCache: SignatureCache): string {
	if (!line.startsWith('data: ')) return line;
	const dataStr = line.substring(6).trim();
	if (!dataStr || dataStr === '[DONE]' || !dataStr.startsWith('{')) return line;
	try {
		const parsed = JSON.parse(dataStr);
		patchResponseBody(parsed, signatureCache);
		return 'data: ' + JSON.stringify(parsed);
	} catch (e) {
		console.error('[patchSSELine] JSON parse failed:', line.substring(0, 200), e);
		return line;
	}
}

/** A Durable Object's behavior is defined in an exported Javascript class */
export class LoadBalancer extends DurableObject {
	env: Env;
	signatureCache: SignatureCache;
	/**
	 * The constructor is invoked once upon creation of the Durable Object, i.e. the first call to
	 * 	`DurableObjectStub::get` for a given identifier (no-op constructors can be omitted)
	 *
	 * @param ctx - The interface for interacting with Durable Object state
	 * @param env - The interface to reference bindings declared in wrangler.jsonc
	 */
	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		this.env = env;
		this.signatureCache = new Map();
		// Initialize the database schema upon first creation.
		this.ctx.storage.sql.exec(`
			CREATE TABLE IF NOT EXISTS api_keys (
				api_key TEXT PRIMARY KEY
			);
			CREATE TABLE IF NOT EXISTS api_key_statuses (
				api_key TEXT PRIMARY KEY,
				status TEXT CHECK(status IN ('normal', 'abnormal')) NOT NULL DEFAULT 'normal',
				last_checked_at INTEGER,
				failed_count INTEGER NOT NULL DEFAULT 0,
				key_group TEXT CHECK(key_group IN ('normal', 'abnormal')) NOT NULL DEFAULT 'normal',
				FOREIGN KEY(api_key) REFERENCES api_keys(api_key) ON DELETE CASCADE
			);
		`);
		this.ctx.storage.setAlarm(Date.now() + 5 * 60 * 1000); // Set an alarm to run in 5 minutes
	}

	async alarm() {
		// 1. Handle abnormal keys
		const abnormalKeys = await this.ctx.storage.sql
			.exec("SELECT api_key, failed_count FROM api_key_statuses WHERE key_group = 'abnormal'")
			.raw<any>();

		for (const row of Array.from(abnormalKeys)) {
			const apiKey = row[0] as string;
			const failedCount = row[1] as number;

			try {
				const response = await fetch(`${BASE_URL}/${API_VERSION}/models/gemini-2.5-flash:generateContent?key=${apiKey}`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
					},
					body: JSON.stringify({
						contents: [{ parts: [{ text: 'hi' }] }],
					}),
				});
				if (response.ok) {
					// Key is working again, move it back to the normal group
					await this.ctx.storage.sql.exec(
						"UPDATE api_key_statuses SET key_group = 'normal', failed_count = 0, last_checked_at = ? WHERE api_key = ?",
						Date.now(),
						apiKey
					);
				} else if (response.status === 429) {
					// Still getting 429, increment failed_count
					const newFailedCount = failedCount + 1;
					if (newFailedCount >= 5) {
						// Delete the key if it has failed 5 times
						await this.ctx.storage.sql.exec('DELETE FROM api_keys WHERE api_key = ?', apiKey);
					} else {
						await this.ctx.storage.sql.exec(
							'UPDATE api_key_statuses SET failed_count = ?, last_checked_at = ? WHERE api_key = ?',
							newFailedCount,
							Date.now(),
							apiKey
						);
					}
				}
			} catch (e) {
				console.error(`Error checking abnormal key ${apiKey}:`, e);
			}
		}

		// 2. Handle normal keys
		const twelveHoursAgo = Date.now() - 12 * 60 * 60 * 1000;
		const normalKeys = await this.ctx.storage.sql
			.exec(
				"SELECT api_key FROM api_key_statuses WHERE key_group = 'normal' AND (last_checked_at IS NULL OR last_checked_at < ?)",
				twelveHoursAgo
			)
			.raw<any>();

		for (const row of Array.from(normalKeys)) {
			const apiKey = row[0] as string;
			try {
				const response = await fetch(`${BASE_URL}/${API_VERSION}/models/gemini-2.5-flash:generateContent?key=${apiKey}`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
					},
					body: JSON.stringify({
						contents: [{ parts: [{ text: 'hi' }] }],
					}),
				});
				if (response.status === 429) {
					// Move to abnormal group
					await this.ctx.storage.sql.exec(
						"UPDATE api_key_statuses SET key_group = 'abnormal', failed_count = 1, last_checked_at = ? WHERE api_key = ?",
						Date.now(),
						apiKey
					);
				} else {
					// Update last_checked_at
					await this.ctx.storage.sql.exec('UPDATE api_key_statuses SET last_checked_at = ? WHERE api_key = ?', Date.now(), apiKey);
				}
			} catch (e) {
				console.error(`Error checking normal key ${apiKey}:`, e);
			}
		}

		// Reschedule the alarm
		this.ctx.storage.setAlarm(Date.now() + 5 * 60 * 1000);
	}

	async fetch(request: Request): Promise<Response> {
		console.log(`[fetch] ${request.method} ${request.url}`);
		if (request.method === 'OPTIONS') {
			return new Response(null, {
				status: 204,
				headers: fixCors({}).headers,
			});
		}
		const url = new URL(request.url);
		const pathname = url.pathname;

		// 静态资源直接放行
		if (pathname === '/favicon.ico' || pathname === '/robots.txt') {
			return new Response('', { status: 204 });
		}

		// 管理 API 权限校验（使用 HOME_ACCESS_KEY）
		if (pathname === '/api/keys' || pathname === '/api/keys/check') {
			if (!isAdminAuthenticated(request, this.env.HOME_ACCESS_KEY)) {
				return new Response(JSON.stringify({ error: 'Unauthorized' }), {
					status: 401,
					headers: fixCors({ headers: { 'Content-Type': 'application/json' } }).headers,
				});
			}
			if (pathname === '/api/keys' && request.method === 'POST') {
				return this.handleApiKeys(request);
			}
			if (pathname === '/api/keys' && request.method === 'GET') {
				return this.getAllApiKeys(request);
			}
			if (pathname === '/api/keys' && request.method === 'DELETE') {
				return this.handleDeleteApiKeys(request);
			}
			if (pathname === '/api/keys/check' && request.method === 'POST') {
				return this.handleApiKeysCheck(request);
			}
		}

		const search = url.search;

		// OpenAI 兼容路由 → 直接 proxy 到 Google 官方 OpenAI 兼容端点
		if (
			pathname.endsWith('/chat/completions') ||
			pathname.endsWith('/completions') ||
			pathname.endsWith('/embeddings') ||
			pathname.endsWith('/v1/models')
		) {
			return this.handleOpenAIProxy(request);
		}

		// OpenAI Responses API route → 仍需手写转换（Google 不支持）
		if (pathname.endsWith('/responses') || pathname.match(/\/responses\/[^/]+$/)) {
			return this.handleResponsesAPI(request);
		}

		// Direct Gemini proxy (原生 Gemini API 请求直接透传)
		const authKey = this.env.AUTH_KEY;
		let targetUrl = `${BASE_URL}${pathname}${search}`;

		if (isForwardClientKeyEnabled(this.env.FORWARD_CLIENT_KEY_ENABLED)) {
			return this.forwardRequestWithLoadBalancing(targetUrl, request);
		}

		// 传统模式：验证 AUTH_KEY
		if (!authKey) {
			return new Response('AUTH_KEY is not configured. Set it in Cloudflare environment variables or local .dev.vars.', {
				status: 500,
				headers: fixCors({}).headers,
			});
		}
		let isAuthorized = false;
		if (search.includes('key=')) {
			const urlObj = new URL(targetUrl);
			const requestKey = urlObj.searchParams.get('key');
			if (requestKey && requestKey === authKey) {
				isAuthorized = true;
			}
		} else {
			const requestKey = request.headers.get('x-goog-api-key');
			if (requestKey && requestKey === authKey) {
				isAuthorized = true;
			}
		}

		if (!isAuthorized) {
			return new Response('Unauthorized', { status: 401, headers: fixCors({}).headers });
		}
		return this.forwardRequestWithLoadBalancing(targetUrl, request);
	}

	// =================================================================================================
	// 通用请求转发（负载均衡）
	// =================================================================================================

	private async forwardRequest(targetUrl: string, request: Request, headers: Headers, apiKey: string): Promise<Response> {
		const response = await fetch(targetUrl, {
			method: request.method,
			headers: headers,
			body: request.method === 'GET' || request.method === 'HEAD' ? null : request.body,
		});

		if (response.status === 429) {
			console.error(`API key ${apiKey} received 429 status code.`);
			await this.ctx.storage.sql.exec(
				"UPDATE api_key_statuses SET key_group = 'abnormal', failed_count = failed_count + 1, last_checked_at = ? WHERE api_key = ?",
				Date.now(),
				apiKey
			);
		}

		const responseHeaders = new Headers(response.headers);
		responseHeaders.set('Access-Control-Allow-Origin', '*');
		responseHeaders.delete('transfer-encoding');
		responseHeaders.delete('connection');
		responseHeaders.delete('keep-alive');
		responseHeaders.delete('content-encoding');
		responseHeaders.set('Referrer-Policy', 'no-referrer');

		return new Response(response.body, {
			status: response.status,
			headers: responseHeaders,
		});
	}

	private async forwardRequestWithLoadBalancing(targetUrl: string, request: Request): Promise<Response> {
		try {
			let headers = new Headers();
			const url = new URL(targetUrl);

			if (request.headers.has('content-type')) {
				headers.set('content-type', request.headers.get('content-type')!);
			}

			if (isForwardClientKeyEnabled(this.env.FORWARD_CLIENT_KEY_ENABLED)) {
				const clientApiKey = this.extractClientApiKey(request, url);
				if (clientApiKey) {
					url.searchParams.set('key', clientApiKey);
					headers.set('x-goog-api-key', clientApiKey);
				}
				return this.forwardRequest(url.toString(), request, headers, clientApiKey || '');
			}

			const apiKey = await this.getRandomApiKey();
			if (!apiKey) {
				return new Response('No API keys configured in the load balancer.', { status: 500 });
			}

			url.searchParams.set('key', apiKey);
			headers.set('x-goog-api-key', apiKey);
			return this.forwardRequest(url.toString(), request, headers, apiKey);
		} catch (error) {
			console.error('Failed to fetch:', error);
			return new Response('Internal Server Error\n' + error, {
				status: 500,
				headers: { 'Content-Type': 'text/plain' },
			});
		}
	}

	// =================================================================================================
	// OpenAI 兼容路由 → 纯 proxy 透传到 Google 官方 OpenAI 兼容端点
	// =================================================================================================

	/**
	 * 将 OpenAI 格式请求直接 proxy 到 Google 的 OpenAI 兼容端点。
	 * Google 端点原生理解 OpenAI 格式，但有少量兼容性问题需要补丁：
	 * - tool_calls 缺少 index 字段（Vercel AI SDK 校验要求必须有）
	 *
	 * 支持: chat/completions, embeddings, models
	 */
	private async handleOpenAIProxy(request: Request): Promise<Response> {
		console.log('[handleOpenAIProxy] Starting');
		const apiKey = await this.resolveApiKey(request);
		if (apiKey instanceof Response) return apiKey;

		const url = new URL(request.url);
		const pathname = url.pathname;
		console.log(`[handleOpenAIProxy] pathname: ${pathname}, method: ${request.method}`);

		// 映射路径到 Google OpenAI 兼容端点
		let targetPath: string;
		let isChatCompletions = false;
		if (pathname.endsWith('/chat/completions')) {
			targetPath = '/chat/completions';
			isChatCompletions = true;
		} else if (pathname.endsWith('/embeddings')) {
			targetPath = '/embeddings';
		} else if (pathname.endsWith('/models')) {
			targetPath = '/models';
		} else {
			return new Response('Not Found', { status: 404 });
		}

		const targetUrl = `${OPENAI_COMPAT_BASE}${targetPath}`;
		console.log(`[handleOpenAIProxy] targetUrl: ${targetUrl}`);

		// 对于 chat/completions，需要读取请求体：
		// 1. 判断是否 streaming
		// 2. 注入缓存的 thought_signature（多轮 tool calling 必须）
		let requestBody: string | null = null;
		let isStreaming = false;
		if (isChatCompletions && request.method === 'POST') {
			requestBody = await request.text();
			try {
				const parsed = JSON.parse(requestBody);
				isStreaming = parsed.stream === true;
				console.log(`[handleOpenAIProxy] model: ${parsed.model}, stream: ${isStreaming}, has_tools: ${!!parsed.tools}, messages_count: ${parsed.messages?.length}`);

				// 将缓存的 thought_signature 注入回 assistant 的 tool_calls
				cleanupSignatureCache(this.signatureCache);
				injectSignaturesIntoRequest(parsed, this.signatureCache);
				requestBody = JSON.stringify(parsed);
			} catch (e) {
				console.error('[handleOpenAIProxy] Failed to parse request body:', e);
			}
		}

		const headers = new Headers();
		headers.set('Authorization', `Bearer ${apiKey}`);
		if (request.headers.has('content-type')) {
			headers.set('Content-Type', request.headers.get('content-type')!);
		}

		console.log(`[handleOpenAIProxy] Sending to Google, streaming: ${isStreaming}`);
		const response = await fetch(targetUrl, {
			method: request.method,
			headers,
			body: requestBody ?? (request.method === 'GET' || request.method === 'HEAD' ? null : request.body),
		});
		console.log(`[handleOpenAIProxy] Google response: ${response.status} ${response.statusText}`);

		// 处理 429 错误：标记 key 异常
		if (response.status === 429) {
			console.error(`[handleOpenAIProxy] API key received 429`);
			await this.ctx.storage.sql.exec(
				"UPDATE api_key_statuses SET key_group = 'abnormal', failed_count = failed_count + 1, last_checked_at = ? WHERE api_key = ?",
				Date.now(),
				apiKey
			);
		}

		// 非 OK 响应：读取错误体并记录日志
		if (!response.ok) {
			const errorBody = await response.text();
			console.error(`[handleOpenAIProxy] Error response from Google: ${response.status}`, errorBody);
			const responseHeaders = new Headers(response.headers);
			responseHeaders.set('Access-Control-Allow-Origin', '*');
			responseHeaders.delete('transfer-encoding');
			responseHeaders.delete('connection');
			responseHeaders.delete('keep-alive');
			responseHeaders.delete('content-encoding');
			responseHeaders.set('Referrer-Policy', 'no-referrer');
			return new Response(errorBody, { status: response.status, headers: responseHeaders });
		}

		const responseHeaders = new Headers(response.headers);
		responseHeaders.set('Access-Control-Allow-Origin', '*');
		responseHeaders.delete('transfer-encoding');
		responseHeaders.delete('connection');
		responseHeaders.delete('keep-alive');
		responseHeaders.delete('content-encoding');
		responseHeaders.set('Referrer-Policy', 'no-referrer');

		// 非 chat/completions 直接透传
		if (!isChatCompletions) {
			return new Response(response.body, { status: response.status, headers: responseHeaders });
		}

		// chat/completions 响应补丁：
		// 1. 补 tool_calls 缺失的 index
		// 2. 提取 thought_signature 并缓存（供后续多轮对话回传）
		// 3. 移除 extra_content（SDK 不认识）
		if (isStreaming) {
			console.log('[handleOpenAIProxy] Applying streaming patch transform');
			const patchedBody = response
				.body!.pipeThrough(new TextDecoderStream())
				.pipeThrough(new TransformStream(createPatchTransform(this.signatureCache)))
				.pipeThrough(new TextEncoderStream());

			return new Response(patchedBody, { status: response.status, headers: responseHeaders });
		} else {
			const body = await response.json();
			console.log('[handleOpenAIProxy] Non-stream response, patching');
			patchResponseBody(body, this.signatureCache);
			return new Response(JSON.stringify(body), { status: response.status, headers: responseHeaders });
		}
	}

	// =================================================================================================
	// 认证 + API Key 解析（公共方法）
	// =================================================================================================

	/**
	 * 从请求中解析出真正要使用的 Gemini API Key。
	 * - FORWARD_CLIENT_KEY_ENABLED 模式：直接用客户端传来的 key
	 * - 传统模式：验证 AUTH_KEY，然后从 key 池中随机选一个
	 *
	 * @returns apiKey string 或 Response（错误响应）
	 */
	private async resolveApiKey(request: Request): Promise<string | Response> {
		const authHeader = request.headers.get('Authorization');
		const clientKey = authHeader?.replace('Bearer ', '') ?? null;

		if (!clientKey) {
			return new Response('No API key found in the client headers, please check your request!', {
				status: 400,
				headers: fixCors({}).headers,
			});
		}

		if (isForwardClientKeyEnabled(this.env.FORWARD_CLIENT_KEY_ENABLED)) {
			return clientKey;
		}

		// 传统模式：验证 AUTH_KEY
		const authKey = this.env.AUTH_KEY;
		if (!authKey) {
			return new Response('AUTH_KEY is not configured. Set it in Cloudflare environment variables or local .dev.vars.', {
				status: 500,
				headers: fixCors({}).headers,
			});
		}
		if (authKey && clientKey !== authKey) {
			return new Response('Unauthorized', { status: 401, headers: fixCors({}).headers });
		}

		// 从 key 池中随机选一个
		const apiKey = await this.getRandomApiKey();
		if (!apiKey) {
			return new Response('No API keys configured in the load balancer.', { status: 500, headers: fixCors({}).headers });
		}
		return apiKey;
	}

	// =================================================================================================
	// OpenAI Responses API（Google 不支持，需要手写转换）
	// =================================================================================================

	private async handleResponsesAPI(request: Request): Promise<Response> {
		const apiKey = await this.resolveApiKey(request);
		if (apiKey instanceof Response) return apiKey;

		const url = new URL(request.url);
		const pathname = url.pathname;

		const errHandler = (err: Error) => {
			console.error('[handleResponsesAPI] Error:', err);
			const errorResponse = {
				error: {
					message: err.message ?? 'Internal Server Error',
					type: 'server_error',
					code: 'internal_error',
				},
			};
			return new Response(JSON.stringify(errorResponse), {
				...fixCors({ headers: { 'Content-Type': 'application/json' } }),
				status: 500,
			});
		};

		try {
			// POST /v1/responses - Create a response
			if (request.method === 'POST' && pathname.endsWith('/responses')) {
				const reqBody = await request.json();
				return this.handleCreateResponse(reqBody, apiKey).catch(errHandler);
			}

			// GET /v1/responses/{id} - Not implemented (stateless proxy)
			if (request.method === 'GET' && pathname.match(/\/responses\/[^/]+$/)) {
				return new Response(
					JSON.stringify({
						error: {
							message: 'This proxy does not support stateful response retrieval. Use store: false in your requests.',
							type: 'invalid_request_error',
							code: 'not_implemented',
						},
					}),
					{ ...fixCors({ headers: { 'Content-Type': 'application/json' } }), status: 501 }
				);
			}

			// DELETE /v1/responses/{id} - Not implemented
			if (request.method === 'DELETE' && pathname.match(/\/responses\/[^/]+$/)) {
				return new Response(
					JSON.stringify({
						error: {
							message: 'This proxy does not support response deletion.',
							type: 'invalid_request_error',
							code: 'not_implemented',
						},
					}),
					{ ...fixCors({ headers: { 'Content-Type': 'application/json' } }), status: 501 }
				);
			}

			return new Response(
				JSON.stringify({
					error: {
						message: 'Method not allowed',
						type: 'invalid_request_error',
						code: 'method_not_allowed',
					},
				}),
				{ ...fixCors({ headers: { 'Content-Type': 'application/json' } }), status: 405 }
			);
		} catch (err: any) {
			return errHandler(err);
		}
	}

	// =================================================================================================
	// Responses API 实现（需要手写转换，因为 Google 官方不支持 Responses API）
	// =================================================================================================

	private generateId(): string {
		const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
		const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
		return Array.from({ length: 29 }, randomChar).join('');
	}

	private async handleCreateResponse(req: any, apiKey: string): Promise<Response> {
		const DEFAULT_MODEL = 'gemini-2.5-flash';
		let model = DEFAULT_MODEL;

		if (typeof req.model === 'string') {
			if (req.model.startsWith('models/')) {
				model = req.model.substring(7);
			} else if (req.model.startsWith('gemini-') || req.model.startsWith('gemma-') || req.model.startsWith('learnlm-')) {
				model = req.model;
			}
		}

		const geminiBody = this.convertResponsesInputToGemini(req);
		const isStreaming = req.stream === true;
		const TASK = isStreaming ? 'streamGenerateContent' : 'generateContent';
		let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
		if (isStreaming) {
			url += '?alt=sse';
		}

		const response = await fetch(url, {
			method: 'POST',
			headers: makeHeaders(apiKey, { 'Content-Type': 'application/json' }),
			body: JSON.stringify(geminiBody),
		});

		if (!response.ok) {
			const errorText = await response.text();
			console.error('[handleCreateResponse] Gemini API error:', response.status, errorText);
			return new Response(
				JSON.stringify({
					error: {
						message: `Gemini API error: ${errorText}`,
						type: 'api_error',
						code: 'upstream_error',
					},
				}),
				{ ...fixCors({ headers: { 'Content-Type': 'application/json' } }), status: response.status }
			);
		}

		const responseId = 'resp_' + this.generateId();
		const messageId = 'msg_' + this.generateId();
		const createdAt = Math.floor(Date.now() / 1000);

		if (isStreaming) {
			return this.handleResponsesStream(response, model, responseId, messageId, createdAt, req);
		} else {
			return this.handleResponsesNonStream(response, model, responseId, messageId, createdAt, req);
		}
	}

	private convertResponsesInputToGemini(req: any): any {
		const harmCategory = [
			'HARM_CATEGORY_HATE_SPEECH',
			'HARM_CATEGORY_SEXUALLY_EXPLICIT',
			'HARM_CATEGORY_DANGEROUS_CONTENT',
			'HARM_CATEGORY_HARASSMENT',
			'HARM_CATEGORY_CIVIC_INTEGRITY',
		];

		const safetySettings = harmCategory.map((category) => ({
			category,
			threshold: 'BLOCK_NONE',
		}));

		const contents: any[] = [];

		// Handle instructions (system message)
		let systemInstruction: any = undefined;
		if (req.instructions) {
			systemInstruction = {
				parts: [{ text: req.instructions }],
			};
		}

		// Handle input - can be string or array
		if (typeof req.input === 'string') {
			contents.push({
				role: 'user',
				parts: [{ text: req.input }],
			});
		} else if (Array.isArray(req.input)) {
			let pendingFunctionCalls: any[] = [];
			let pendingFunctionResponses: any[] = [];

			for (const item of req.input) {
				if (item.type === 'function_call') {
					if (pendingFunctionResponses.length > 0) {
						contents.push({ role: 'user', parts: pendingFunctionResponses });
						pendingFunctionResponses = [];
					}
					let args = item.arguments ?? {};
					if (typeof args === 'string') {
						try { args = JSON.parse(args); } catch { args = {}; }
					}
					pendingFunctionCalls.push({
						functionCall: { name: item.name, args },
					});
					continue;
				}

				if (item.type === 'function_call_output') {
					if (pendingFunctionCalls.length > 0) {
						contents.push({ role: 'model', parts: pendingFunctionCalls });
						pendingFunctionCalls = [];
					}
					let output = item.output;
					if (typeof output === 'string') {
						try { output = JSON.parse(output); } catch { output = { result: output }; }
					}
					pendingFunctionResponses.push({
						functionResponse: {
							name: item.call_id ? item.call_id.split('_')[1] : 'unknown',
							response: output,
						},
					});
					continue;
				}

				// Flush pending items before other types
				if (pendingFunctionCalls.length > 0) {
					contents.push({ role: 'model', parts: pendingFunctionCalls });
					pendingFunctionCalls = [];
				}
				if (pendingFunctionResponses.length > 0) {
					contents.push({ role: 'user', parts: pendingFunctionResponses });
					pendingFunctionResponses = [];
				}

				if (item.type === 'message') {
					const role = item.role === 'assistant' ? 'model' : 'user';
					const parts: any[] = [];

					if (Array.isArray(item.content)) {
						for (const contentPart of item.content) {
							if (contentPart.type === 'input_text' || contentPart.type === 'output_text') {
								parts.push({ text: contentPart.text });
							} else if (contentPart.type === 'input_image') {
								if (contentPart.image_url) {
									parts.push({
										inlineData: {
											mimeType: 'image/jpeg',
											data: contentPart.image_url.replace(/^data:image\/[^;]+;base64,/, ''),
										},
									});
								}
							}
						}
					} else if (typeof item.content === 'string') {
						parts.push({ text: item.content });
					}

					if (parts.length > 0) {
						contents.push({ role, parts });
					}
				} else if (item.role) {
					const role = item.role === 'assistant' ? 'model' : 'user';
					const parts: any[] = [];

					if (Array.isArray(item.content)) {
						for (const contentPart of item.content) {
							if (typeof contentPart === 'string') {
								parts.push({ text: contentPart });
							} else if (contentPart.type === 'text' || contentPart.type === 'input_text') {
								parts.push({ text: contentPart.text });
							}
						}
					} else if (typeof item.content === 'string') {
						parts.push({ text: item.content });
					}

					if (parts.length > 0) {
						contents.push({ role, parts });
					}
				}
			}

			// Flush remaining
			if (pendingFunctionCalls.length > 0) {
				contents.push({ role: 'model', parts: pendingFunctionCalls });
			}
			if (pendingFunctionResponses.length > 0) {
				contents.push({ role: 'user', parts: pendingFunctionResponses });
			}
		}

		// Build generation config
		const generationConfig: any = {};
		if (req.temperature !== undefined) generationConfig.temperature = req.temperature;
		if (req.top_p !== undefined) generationConfig.topP = req.top_p;
		if (req.max_output_tokens !== undefined) generationConfig.maxOutputTokens = req.max_output_tokens;

		// Handle function tools
		const tools: any[] = [];
		if (Array.isArray(req.tools)) {
			const functionDeclarations: any[] = [];
			for (const tool of req.tools) {
				if (tool.type === 'function') {
					const params = tool.parameters ? { ...tool.parameters } : undefined;
					if (params) {
						delete params.$schema;
						delete params.additionalProperties;
					}
					functionDeclarations.push({
						name: tool.name,
						description: tool.description,
						parameters: params,
					});
				}
			}
			if (functionDeclarations.length > 0) {
				tools.push({ function_declarations: functionDeclarations });
			}
		}

		// Handle tool_choice
		let toolConfig: any = undefined;
		if (req.tool_choice) {
			if (typeof req.tool_choice === 'string') {
				if (req.tool_choice === 'required') {
					toolConfig = { function_calling_config: { mode: 'ANY' } };
				} else if (req.tool_choice === 'none') {
					toolConfig = { function_calling_config: { mode: 'NONE' } };
				}
			} else if (req.tool_choice.type === 'function') {
				toolConfig = {
					function_calling_config: {
						mode: 'ANY',
						allowed_function_names: [req.tool_choice.name],
					},
				};
			}
		}

		return {
			contents,
			safetySettings,
			generationConfig,
			...(systemInstruction ? { system_instruction: systemInstruction } : {}),
			...(tools.length > 0 ? { tools } : {}),
			...(toolConfig ? { tool_config: toolConfig } : {}),
		};
	}

	private async handleResponsesNonStream(
		response: Response,
		model: string,
		responseId: string,
		messageId: string,
		createdAt: number,
		req: any
	): Promise<Response> {
		let geminiResponse: any;
		try {
			geminiResponse = await response.json();
		} catch (err) {
			console.error('[handleResponsesNonStream] Failed to parse Gemini response:', err);
			return new Response(
				JSON.stringify({
					error: {
						message: 'Failed to parse Gemini response',
						type: 'api_error',
						code: 'parse_error',
					},
				}),
				{ ...fixCors({ headers: { 'Content-Type': 'application/json' } }), status: 500 }
			);
		}

		let outputText = '';
		const functionCalls: any[] = [];
		const candidate = geminiResponse.candidates?.[0];

		if (candidate?.content?.parts) {
			for (const part of candidate.content.parts) {
				if (part.text) {
					outputText += part.text;
				}
				if (part.functionCall) {
					const callId = `call_${part.functionCall.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
					functionCalls.push({
						type: 'function_call',
						id: callId,
						call_id: callId,
						name: part.functionCall.name,
						arguments: JSON.stringify(part.functionCall.args || {}),
						status: 'completed',
					});
				}
			}
		}

		const usageMetadata = geminiResponse.usageMetadata || {};
		const inputTokens = usageMetadata.promptTokenCount || 0;
		const outputTokens = usageMetadata.candidatesTokenCount || 0;

		const output: any[] = [];
		for (const fc of functionCalls) {
			output.push(fc);
		}

		if (outputText || functionCalls.length === 0) {
			output.push({
				type: 'message',
				id: messageId,
				status: 'completed',
				role: 'assistant',
				content: [
					{
						type: 'output_text',
						text: outputText,
						annotations: [],
					},
				],
			});
		}

		const openAIResponse = {
			id: responseId,
			object: 'response',
			created_at: createdAt,
			status: 'completed',
			completed_at: Math.floor(Date.now() / 1000),
			error: null,
			incomplete_details: null,
			instructions: req.instructions || null,
			max_output_tokens: req.max_output_tokens || null,
			model: model,
			output: output,
			parallel_tool_calls: req.parallel_tool_calls ?? true,
			previous_response_id: req.previous_response_id || null,
			reasoning: { effort: null, summary: null },
			store: req.store ?? true,
			temperature: req.temperature ?? 1.0,
			text: { format: { type: 'text' } },
			tool_choice: req.tool_choice ?? 'auto',
			tools: req.tools || [],
			top_p: req.top_p ?? 1.0,
			truncation: req.truncation ?? 'disabled',
			usage: {
				input_tokens: inputTokens,
				input_tokens_details: { cached_tokens: 0 },
				output_tokens: outputTokens,
				output_tokens_details: { reasoning_tokens: 0 },
				total_tokens: inputTokens + outputTokens,
			},
			user: req.user || null,
			metadata: req.metadata || {},
		};

		return new Response(JSON.stringify(openAIResponse), {
			...fixCors({ headers: { 'Content-Type': 'application/json' } }),
		});
	}

	private handleResponsesStream(
		response: Response,
		model: string,
		responseId: string,
		messageId: string,
		createdAt: number,
		req: any
	): Response {
		const encoder = new TextEncoder();
		let sequenceNumber = 0;

		const buildResponseSnapshot = (overrides: any) => ({
			id: responseId,
			object: 'response',
			created_at: createdAt,
			error: null,
			incomplete_details: null,
			instructions: req.instructions || null,
			max_output_tokens: req.max_output_tokens || null,
			model: model,
			output: [],
			parallel_tool_calls: req.parallel_tool_calls ?? true,
			previous_response_id: req.previous_response_id || null,
			reasoning: { effort: null, summary: null },
			store: req.store ?? true,
			temperature: req.temperature ?? 1.0,
			text: { format: { type: 'text' } },
			tool_choice: req.tool_choice ?? 'auto',
			tools: req.tools || [],
			top_p: req.top_p ?? 1.0,
			truncation: req.truncation ?? 'disabled',
			usage: null,
			user: req.user || null,
			metadata: req.metadata || {},
			...overrides,
		});

		const stream = new ReadableStream({
			async start(controller) {
				const sendEvent = (data: any) => {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
				};

				// 1. response.created
				sendEvent({
					type: 'response.created',
					response: buildResponseSnapshot({ status: 'in_progress', completed_at: null }),
					sequence_number: ++sequenceNumber,
				});

				// 2. response.in_progress
				sendEvent({
					type: 'response.in_progress',
					response: buildResponseSnapshot({ status: 'in_progress', completed_at: null }),
					sequence_number: ++sequenceNumber,
				});

				// Process Gemini SSE stream
				const reader = response.body!.getReader();
				const decoder = new TextDecoder();
				let buffer = '';
				let fullText = '';
				let inputTokens = 0;
				let outputTokens = 0;

				const functionCalls: any[] = [];
				const seenFunctionCalls = new Set<string>();
				let outputIndex = 0;
				let textOutputIndex = -1;
				let textContentStarted = false;

				try {
					while (true) {
						const { done, value } = await reader.read();
						if (done) break;

						buffer += decoder.decode(value, { stream: true });
						const lines = buffer.split('\n');
						buffer = lines.pop() || '';

						for (const line of lines) {
							if (line.startsWith('data: ')) {
								const dataStr = line.substring(6).trim();
								if (!dataStr || dataStr === '[DONE]') continue;

								try {
									const data = JSON.parse(dataStr);
									const candidate = data.candidates?.[0];

									if (candidate?.content?.parts) {
										for (const part of candidate.content.parts) {
											// Handle function calls
											if (part.functionCall) {
												const funcName = part.functionCall.name;
												const funcArgs = part.functionCall.args || {};
												const callId = `call_${funcName}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

												const callKey = `${funcName}_${JSON.stringify(funcArgs)}`;
												if (!seenFunctionCalls.has(callKey)) {
													seenFunctionCalls.add(callKey);

													const funcCallItem = {
														type: 'function_call',
														id: callId,
														call_id: callId,
														name: funcName,
														arguments: JSON.stringify(funcArgs),
														status: 'completed',
													};
													functionCalls.push(funcCallItem);

													sendEvent({
														type: 'response.output_item.added',
														output_index: outputIndex,
														item: {
															type: 'function_call',
															id: callId,
															call_id: callId,
															name: funcName,
															arguments: '',
															status: 'in_progress',
														},
														sequence_number: ++sequenceNumber,
													});

													sendEvent({
														type: 'response.function_call_arguments.delta',
														item_id: callId,
														output_index: outputIndex,
														delta: JSON.stringify(funcArgs),
														sequence_number: ++sequenceNumber,
													});

													sendEvent({
														type: 'response.function_call_arguments.done',
														item_id: callId,
														output_index: outputIndex,
														arguments: JSON.stringify(funcArgs),
														sequence_number: ++sequenceNumber,
													});

													sendEvent({
														type: 'response.output_item.done',
														output_index: outputIndex,
														item: funcCallItem,
														sequence_number: ++sequenceNumber,
													});

													outputIndex++;
												}
											}

											// Handle text content
											if (part.text) {
												if (!textContentStarted) {
													textContentStarted = true;
													textOutputIndex = outputIndex;
													outputIndex++;

													sendEvent({
														type: 'response.output_item.added',
														output_index: textOutputIndex,
														item: {
															id: messageId,
															status: 'in_progress',
															type: 'message',
															role: 'assistant',
															content: [],
														},
														sequence_number: ++sequenceNumber,
													});

													sendEvent({
														type: 'response.content_part.added',
														item_id: messageId,
														output_index: textOutputIndex,
														content_index: 0,
														part: {
															type: 'output_text',
															text: '',
															annotations: [],
														},
														sequence_number: ++sequenceNumber,
													});
												}

												fullText += part.text;

												sendEvent({
													type: 'response.output_text.delta',
													item_id: messageId,
													output_index: textOutputIndex,
													content_index: 0,
													delta: part.text,
													sequence_number: ++sequenceNumber,
												});
											}
										}
									}

									if (data.usageMetadata) {
										inputTokens = data.usageMetadata.promptTokenCount || inputTokens;
										outputTokens = data.usageMetadata.candidatesTokenCount || outputTokens;
									}
								} catch (parseErr) {
									console.error('[handleResponsesStream] Failed to parse SSE data:', parseErr);
								}
							}
						}
					}
				} catch (streamErr) {
					console.error('[handleResponsesStream] Stream error:', streamErr);
				}

				const finalOutput: any[] = [];
				for (const fc of functionCalls) {
					finalOutput.push(fc);
				}

				if (textContentStarted) {
					sendEvent({
						type: 'response.output_text.done',
						item_id: messageId,
						output_index: textOutputIndex,
						content_index: 0,
						text: fullText,
						sequence_number: ++sequenceNumber,
					});

					sendEvent({
						type: 'response.content_part.done',
						item_id: messageId,
						output_index: textOutputIndex,
						content_index: 0,
						part: { type: 'output_text', text: fullText, annotations: [] },
						sequence_number: ++sequenceNumber,
					});

					sendEvent({
						type: 'response.output_item.done',
						output_index: textOutputIndex,
						item: {
							id: messageId,
							status: 'completed',
							type: 'message',
							role: 'assistant',
							content: [{ type: 'output_text', text: fullText, annotations: [] }],
						},
						sequence_number: ++sequenceNumber,
					});

					finalOutput.push({
						type: 'message',
						id: messageId,
						status: 'completed',
						role: 'assistant',
						content: [{ type: 'output_text', text: fullText, annotations: [] }],
					});
				} else if (functionCalls.length === 0) {
					sendEvent({
						type: 'response.output_item.added',
						output_index: 0,
						item: { id: messageId, status: 'completed', type: 'message', role: 'assistant', content: [] },
						sequence_number: ++sequenceNumber,
					});

					sendEvent({
						type: 'response.output_item.done',
						output_index: 0,
						item: { id: messageId, status: 'completed', type: 'message', role: 'assistant', content: [] },
						sequence_number: ++sequenceNumber,
					});

					finalOutput.push({
						type: 'message',
						id: messageId,
						status: 'completed',
						role: 'assistant',
						content: [],
					});
				}

				const completedAt = Math.floor(Date.now() / 1000);
				sendEvent({
					type: 'response.completed',
					response: buildResponseSnapshot({
						status: 'completed',
						completed_at: completedAt,
						output: finalOutput,
						usage: {
							input_tokens: inputTokens,
							input_tokens_details: { cached_tokens: 0 },
							output_tokens: outputTokens,
							output_tokens_details: { reasoning_tokens: 0 },
							total_tokens: inputTokens + outputTokens,
						},
					}),
					sequence_number: ++sequenceNumber,
				});

				controller.close();
			},
		});

		return new Response(stream, {
			headers: {
				...fixCors({}).headers,
				'Content-Type': 'text/event-stream',
				'Cache-Control': 'no-cache',
				Connection: 'keep-alive',
			},
		});
	}

	// =================================================================================================
	// Admin API Handlers
	// =================================================================================================

	async handleApiKeys(request: Request): Promise<Response> {
		try {
			const { keys } = (await request.json()) as { keys: string[] };
			if (!Array.isArray(keys) || keys.length === 0) {
				return new Response(JSON.stringify({ error: '请求体无效，需要一个包含key的非空数组。' }), {
					status: 400,
					headers: { 'Content-Type': 'application/json' },
				});
			}

			for (const key of keys) {
				await this.ctx.storage.sql.exec('INSERT OR IGNORE INTO api_keys (api_key) VALUES (?)', key);
				await this.ctx.storage.sql.exec('INSERT OR IGNORE INTO api_key_statuses (api_key) VALUES (?)', key);
			}

			return new Response(JSON.stringify({ message: 'API密钥添加成功。' }), {
				status: 200,
				headers: { 'Content-Type': 'application/json' },
			});
		} catch (error: any) {
			console.error('处理API密钥失败:', error);
			return new Response(JSON.stringify({ error: error.message || '内部服务器错误' }), {
				status: 500,
				headers: { 'Content-Type': 'application/json' },
			});
		}
	}

	async handleDeleteApiKeys(request: Request): Promise<Response> {
		try {
			const { keys } = (await request.json()) as { keys: string[] };
			if (!Array.isArray(keys) || keys.length === 0) {
				return new Response(JSON.stringify({ error: '请求体无效，需要一个包含key的非空数组。' }), {
					status: 400,
					headers: { 'Content-Type': 'application/json' },
				});
			}

			const batchSize = 500;
			for (let i = 0; i < keys.length; i += batchSize) {
				const batch = keys.slice(i, i + batchSize);
				const placeholders = batch.map(() => '?').join(',');
				await this.ctx.storage.sql.exec(`DELETE FROM api_keys WHERE api_key IN (${placeholders})`, ...batch);
			}

			return new Response(JSON.stringify({ message: 'API密钥删除成功。' }), {
				status: 200,
				headers: { 'Content-Type': 'application/json' },
			});
		} catch (error: any) {
			console.error('删除API密钥失败:', error);
			return new Response(JSON.stringify({ error: error.message || '内部服务器错误' }), {
				status: 500,
				headers: { 'Content-Type': 'application/json' },
			});
		}
	}

	async handleApiKeysCheck(request: Request): Promise<Response> {
		try {
			const { keys } = (await request.json()) as { keys: string[] };
			if (!Array.isArray(keys) || keys.length === 0) {
				return new Response(JSON.stringify({ error: '请求体无效，需要一个包含key的非空数组。' }), {
					status: 400,
					headers: { 'Content-Type': 'application/json' },
				});
			}

			const checkResults = await Promise.all(
				keys.map(async (key) => {
					try {
						const response = await fetch(`${BASE_URL}/${API_VERSION}/models/gemini-2.5-flash:generateContent?key=${key}`, {
							method: 'POST',
							headers: { 'Content-Type': 'application/json' },
							body: JSON.stringify({
								contents: [{ parts: [{ text: 'hi' }] }],
							}),
						});
						return { key, valid: response.ok, error: response.ok ? null : await response.text() };
					} catch (e: any) {
						return { key, valid: false, error: e.message };
					}
				})
			);

			for (const result of checkResults) {
				if (result.valid) {
					await this.ctx.storage.sql.exec(
						"UPDATE api_key_statuses SET status = 'normal', key_group = 'normal', failed_count = 0, last_checked_at = ? WHERE api_key = ?",
						Date.now(),
						result.key
					);
				} else {
					await this.ctx.storage.sql.exec(
						"UPDATE api_key_statuses SET status = 'abnormal', key_group = 'abnormal', failed_count = failed_count + 1, last_checked_at = ? WHERE api_key = ?",
						Date.now(),
						result.key
					);
				}
			}

			return new Response(JSON.stringify(checkResults), {
				headers: { 'Content-Type': 'application/json' },
			});
		} catch (error: any) {
			console.error('检查API密钥失败:', error);
			return new Response(JSON.stringify({ error: error.message || '内部服务器错误' }), {
				status: 500,
				headers: { 'Content-Type': 'application/json' },
			});
		}
	}

	async getAllApiKeys(request: Request): Promise<Response> {
		try {
			const url = new URL(request.url);
			const page = parseInt(url.searchParams.get('page') || '1', 10);
			const pageSize = parseInt(url.searchParams.get('pageSize') || '50', 10);
			const offset = (page - 1) * pageSize;

			const totalResult = await this.ctx.storage.sql.exec('SELECT COUNT(*) as count FROM api_key_statuses').raw<any>();
			const totalArray = Array.from(totalResult);
			const total = totalArray.length > 0 ? totalArray[0][0] : 0;

			const results = await this.ctx.storage.sql
				.exec('SELECT api_key, status, key_group, last_checked_at, failed_count FROM api_key_statuses LIMIT ? OFFSET ?', pageSize, offset)
				.raw<any>();
			const keys = results
				? Array.from(results).map((row: any) => ({
						api_key: row[0],
						status: row[1],
						key_group: row[2],
						last_checked_at: row[3],
						failed_count: row[4],
				  }))
				: [];

			return new Response(JSON.stringify({ keys, total }), {
				headers: { 'Content-Type': 'application/json' },
			});
		} catch (error: any) {
			console.error('获取API密钥失败:', error);
			return new Response(JSON.stringify({ error: error.message || '内部服务器错误' }), {
				status: 500,
				headers: { 'Content-Type': 'application/json' },
			});
		}
	}

	// =================================================================================================
	// Helper Methods
	// =================================================================================================

	private extractClientApiKey(request: Request, url: URL): string | null {
		if (url.searchParams.has('key')) {
			const key = url.searchParams.get('key');
			if (key) return key;
		}

		const googApiKey = request.headers.get('x-goog-api-key');
		if (googApiKey) return googApiKey;

		const authHeader = request.headers.get('Authorization');
		if (authHeader && authHeader.startsWith('Bearer ')) {
			return authHeader.substring(7);
		}

		return null;
	}

	private async getRandomApiKey(): Promise<string | null> {
		try {
			let results = await this.ctx.storage.sql
				.exec("SELECT api_key FROM api_key_statuses WHERE key_group = 'normal' ORDER BY RANDOM() LIMIT 1")
				.raw<any>();
			let keys = Array.from(results);
			if (keys && keys.length > 0) {
				return keys[0][0] as string;
			}

			results = await this.ctx.storage.sql
				.exec("SELECT api_key FROM api_key_statuses WHERE key_group = 'abnormal' ORDER BY RANDOM() LIMIT 1")
				.raw<any>();
			keys = Array.from(results);
			if (keys && keys.length > 0) {
				return keys[0][0] as string;
			}

			return null;
		} catch (error) {
			console.error('获取随机API密钥失败:', error);
			return null;
		}
	}
}
