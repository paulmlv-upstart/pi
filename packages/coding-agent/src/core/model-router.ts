import {
	type Api,
	type AssistantMessage,
	type Context,
	type Model,
	type SimpleStreamOptions,
	streamSimple,
} from "@earendil-works/pi-ai";
import { isRouterModel, type ModelRegistry, type RouterConfig } from "./model-registry.ts";
import { findExactModelReferenceMatch, resolveModelScope } from "./model-resolver.ts";

export interface RouteCandidate {
	id: string;
	model: Model<Api>;
	name: string;
	contextWindow: number;
	maxTokens: number;
	input: ("text" | "image")[];
}

export interface RouteDecision {
	routerId: string;
	model: Model<Api>;
	selectorModel: Model<Api>;
	reason: string;
	candidates: RouteCandidate[];
	fallbackUsed: boolean;
}

interface ResolveRouteOptions {
	requestedModel: Model<Api>;
	context: Context;
	streamOptions?: SimpleStreamOptions;
}

interface SelectorChoice {
	model: string;
	reason: string;
}

function canonicalModelId(model: Model<Api>): string {
	return `${model.provider}/${model.id}`;
}

function toRouteCandidate(model: Model<Api>): RouteCandidate {
	return {
		id: canonicalModelId(model),
		model,
		name: model.name,
		contextWindow: model.contextWindow,
		maxTokens: model.maxTokens,
		input: model.input,
	};
}

function getTextFromMessage(message: Context["messages"][number]): string {
	if (message.role === "user") {
		if (typeof message.content === "string") return message.content;
		return message.content
			.filter((content) => content.type === "text")
			.map((content) => content.text)
			.join("\n");
	}
	if (message.role === "assistant") {
		return message.content
			.filter((content) => content.type === "text")
			.map((content) => content.text)
			.join("\n");
	}
	return message.content
		.filter((content) => content.type === "text")
		.map((content) => content.text)
		.join("\n");
}

function truncateText(text: string, maxChars: number): string {
	return text.length <= maxChars ? text : `${text.slice(0, maxChars)}...`;
}

function getCurrentUserRequest(context: Context): string {
	for (let index = context.messages.length - 1; index >= 0; index--) {
		const message = context.messages[index];
		if (message.role === "user") {
			return truncateText(getTextFromMessage(message), 1200);
		}
	}
	return "";
}

function getContextSizeBucket(context: Context): string {
	const text = [context.systemPrompt ?? "", ...context.messages.map((message) => getTextFromMessage(message))].join(
		"\n",
	);
	const estimatedTokens = Math.ceil(text.length / 4);
	if (estimatedTokens < 8000) return "small";
	if (estimatedTokens < 64000) return "medium";
	return "large";
}

function buildSelectorPrompt(context: Context, candidates: RouteCandidate[]): string {
	const candidateLines = candidates
		.map(
			(candidate) =>
				`- ${candidate.id}: ${candidate.name}; context ${candidate.contextWindow}; max output ${candidate.maxTokens}; input ${candidate.input.join(",")}`,
		)
		.join("\n");

	return [
		"Select one concrete model for the next coding-agent turn.",
		"Return exactly one JSON object with fields: model, reason.",
		"Use the model id exactly as listed in Candidates.",
		"",
		`Current user request:\n${getCurrentUserRequest(context) || "(empty)"}`,
		"",
		`Task state: ${context.messages.length > 1 ? "continuation" : "new task"}`,
		`Context size: ${getContextSizeBucket(context)}`,
		`Tools enabled: ${(context.tools?.length ?? 0) > 0 ? "yes" : "no"}`,
		"",
		`Candidates:\n${candidateLines}`,
	].join("\n");
}

function extractJsonObject(text: string): string | undefined {
	const start = text.indexOf("{");
	const end = text.lastIndexOf("}");
	if (start === -1 || end === -1 || end < start) return undefined;
	return text.slice(start, end + 1);
}

function parseSelectorChoice(message: AssistantMessage): SelectorChoice | undefined {
	const text = message.content
		.filter((content) => content.type === "text")
		.map((content) => content.text)
		.join("\n")
		.trim();
	const jsonText = extractJsonObject(text);
	if (!jsonText) return undefined;

	const parsed: unknown = JSON.parse(jsonText);
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return undefined;
	const record = parsed as Record<string, unknown>;
	const model = record.model;
	const reason = record.reason;
	if (typeof model !== "string" || !model.trim()) return undefined;
	return {
		model: model.trim(),
		reason: typeof reason === "string" && reason.trim() ? reason.trim() : "selector returned no reason",
	};
}

function findCandidateByReference(reference: string, candidates: RouteCandidate[]): RouteCandidate | undefined {
	const models = candidates.map((candidate) => candidate.model);
	const exact = findExactModelReferenceMatch(reference, models);
	if (!exact) return undefined;
	return candidates.find(
		(candidate) => candidate.model.provider === exact.provider && candidate.model.id === exact.id,
	);
}

function chooseFallback(config: RouterConfig, candidates: RouteCandidate[]): RouteCandidate {
	if (config.fallback) {
		const fallback = findCandidateByReference(config.fallback, candidates);
		if (fallback) return fallback;
	}
	return candidates[0];
}

export class ModelRouter {
	private readonly modelRegistry: ModelRegistry;

	constructor(modelRegistry: ModelRegistry) {
		this.modelRegistry = modelRegistry;
	}

	async resolve(options: ResolveRouteOptions): Promise<RouteDecision | undefined> {
		if (!isRouterModel(options.requestedModel)) return undefined;

		const routerId = options.requestedModel.id;
		const config = this.modelRegistry.getRouterConfig(routerId);
		if (!config) {
			throw new Error(`Router ${routerId}: configuration not found.`);
		}

		const selectorModel = this.resolveSelectorModel(routerId, config);
		const selectorAuth = await this.modelRegistry.getApiKeyAndHeaders(selectorModel);
		if (!selectorAuth.ok) {
			throw new Error(
				`Router ${routerId}: selector model ${canonicalModelId(selectorModel)} auth failed: ${selectorAuth.error}`,
			);
		}

		const candidates = await this.resolveEligibleCandidates(routerId, config);
		if (candidates.length === 0) {
			throw new Error(`Router ${routerId}: no eligible candidates with configured auth.`);
		}

		try {
			const choice = await this.callSelector(
				selectorModel,
				selectorAuth,
				options.context,
				candidates,
				options.streamOptions,
			);
			const selected = findCandidateByReference(choice.model, candidates);
			if (selected) {
				return {
					routerId,
					model: selected.model,
					selectorModel,
					reason: choice.reason,
					candidates,
					fallbackUsed: false,
				};
			}
		} catch {
			// Selector output and transport failures fall through to configured fallback.
		}

		const fallback = chooseFallback(config, candidates);
		return {
			routerId,
			model: fallback.model,
			selectorModel,
			reason: "router selector failed or returned an invalid model; using fallback",
			candidates,
			fallbackUsed: true,
		};
	}

	private resolveSelectorModel(routerId: string, config: RouterConfig): Model<Api> {
		const selectorModel = findExactModelReferenceMatch(config.selectorModel, this.modelRegistry.getAll());
		if (!selectorModel) {
			throw new Error(`Router ${routerId}: selector model "${config.selectorModel}" was not found.`);
		}
		if (isRouterModel(selectorModel)) {
			throw new Error(`Router ${routerId}: selectorModel must resolve to a concrete model, not a router.`);
		}
		return selectorModel;
	}

	private async resolveEligibleCandidates(routerId: string, config: RouterConfig): Promise<RouteCandidate[]> {
		const scopedCandidates = await resolveModelScope(config.candidates, this.modelRegistry);
		const candidates: RouteCandidate[] = [];

		for (const scopedCandidate of scopedCandidates) {
			const model = scopedCandidate.model;
			if (isRouterModel(model) || !this.modelRegistry.hasConfiguredAuth(model)) continue;
			if (
				candidates.some(
					(candidate) => candidate.model.provider === model.provider && candidate.model.id === model.id,
				)
			) {
				continue;
			}
			candidates.push(toRouteCandidate(model));
		}

		if (candidates.length === 0) {
			const configured =
				this.modelRegistry.getRouterCandidateStatus(routerId)?.configuredCandidates ?? config.candidates.length;
			if (configured === 0) return [];
		}

		return candidates;
	}

	private async callSelector(
		selectorModel: Model<Api>,
		selectorAuth: Extract<Awaited<ReturnType<ModelRegistry["getApiKeyAndHeaders"]>>, { ok: true }>,
		context: Context,
		candidates: RouteCandidate[],
		streamOptions: SimpleStreamOptions | undefined,
	): Promise<SelectorChoice> {
		const selectorContext: Context = {
			systemPrompt:
				"You are a strict routing selector for coding tasks. Output only valid JSON and do not use tools.",
			messages: [
				{
					role: "user",
					content: buildSelectorPrompt(context, candidates),
					timestamp: Date.now(),
				},
			],
		};

		const response = await streamSimple(selectorModel, selectorContext, {
			apiKey: selectorAuth.apiKey,
			headers: selectorAuth.headers,
			maxTokens: 256,
			signal: streamOptions?.signal,
			timeoutMs: streamOptions?.timeoutMs,
			websocketConnectTimeoutMs: streamOptions?.websocketConnectTimeoutMs,
			maxRetries: streamOptions?.maxRetries,
			maxRetryDelayMs: streamOptions?.maxRetryDelayMs,
			cacheRetention: "none",
		}).result();

		if (response.stopReason === "error" || response.stopReason === "aborted") {
			throw new Error(response.errorMessage ?? `Selector stopped with ${response.stopReason}`);
		}

		const choice = parseSelectorChoice(response);
		if (!choice) {
			throw new Error("Selector returned invalid JSON.");
		}
		return choice;
	}
}
