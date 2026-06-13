import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Context, fauxAssistantMessage, registerFauxProvider, type StreamOptions } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { ModelRegistry } from "../src/core/model-registry.ts";
import { ModelRouter } from "../src/core/model-router.ts";

describe("ModelRouter", () => {
	let tempDir: string;
	let modelsJsonPath: string;
	let authStorage: AuthStorage;
	let faux: ReturnType<typeof registerFauxProvider>;

	beforeEach(() => {
		tempDir = mkdtempSync(join(tmpdir(), "pi-model-router-"));
		mkdirSync(tempDir, { recursive: true });
		modelsJsonPath = join(tempDir, "models.json");
		authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		faux = registerFauxProvider({
			provider: "route-provider",
			models: [{ id: "selector" }, { id: "candidate-a" }, { id: "candidate-b" }],
		});
	});

	afterEach(() => {
		faux.unregister();
		rmSync(tempDir, { recursive: true, force: true });
	});

	function writeConfig(
		options: {
			candidates?: string[];
			fallback?: string;
			selectorModel?: string;
			providers?: Record<string, unknown>;
		} = {},
	): void {
		writeFileSync(
			modelsJsonPath,
			JSON.stringify({
				providers: options.providers ?? {
					"route-provider": {
						baseUrl: "https://route.test/v1",
						apiKey: "test-key",
						api: faux.api,
						models: [{ id: "selector" }, { id: "candidate-a" }, { id: "candidate-b" }],
					},
				},
				routers: {
					"upstart-router": {
						selectorModel: options.selectorModel ?? "route-provider/selector",
						candidates: options.candidates ?? ["route-provider/candidate-a", "route-provider/candidate-b"],
						fallback: options.fallback,
					},
				},
			}),
		);
	}

	function createRouterContext(request = "fix the failing tests"): Context {
		return {
			systemPrompt: "system prompt that should not include full history",
			messages: [
				{ role: "user", content: "old request that should not be included", timestamp: 1 },
				{
					role: "assistant",
					content: [{ type: "text", text: "old assistant response that should not be included" }],
					api: "faux",
					provider: "route-provider",
					model: "candidate-a",
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "stop",
					timestamp: 2,
				},
				{ role: "user", content: request, timestamp: 3 },
			],
			tools: [{ name: "read", description: "Read a file", parameters: Type.Object({}) }],
		};
	}

	function createRegistryAndRouter(): { registry: ModelRegistry; router: ModelRouter } {
		const registry = ModelRegistry.create(authStorage, modelsJsonPath);
		return { registry, router: new ModelRouter(registry) };
	}

	function getRouterModel(registry: ModelRegistry) {
		const model = registry.find("router", "upstart-router");
		if (!model) throw new Error("router model not found");
		return model;
	}

	test("expands candidate globs and uses a valid selector response", async () => {
		writeConfig({ candidates: ["route-provider/candidate-*"] });
		faux.setResponses([
			fauxAssistantMessage(
				JSON.stringify({ model: "route-provider/candidate-b", reason: "large multi-file coding task" }),
			),
		]);
		const { registry, router } = createRegistryAndRouter();

		const decision = await router.resolve({
			requestedModel: getRouterModel(registry),
			context: createRouterContext(),
		});

		expect(decision?.model.id).toBe("candidate-b");
		expect(decision?.reason).toBe("large multi-file coding task");
		expect(decision?.fallbackUsed).toBe(false);
		expect(decision?.candidates.map((candidate) => candidate.id)).toEqual([
			"route-provider/candidate-a",
			"route-provider/candidate-b",
		]);
	});

	test("sends compact coding-only input to the selector", async () => {
		writeConfig();
		let selectorPrompt = "";
		faux.setResponses([
			(context) => {
				const message = context.messages[0];
				selectorPrompt = message.role === "user" && typeof message.content === "string" ? message.content : "";
				return fauxAssistantMessage(JSON.stringify({ model: "route-provider/candidate-a", reason: "small task" }));
			},
		]);
		const { registry, router } = createRegistryAndRouter();

		await router.resolve({
			requestedModel: getRouterModel(registry),
			context: createRouterContext("current routing request"),
		});

		expect(selectorPrompt).toContain("current routing request");
		expect(selectorPrompt).toContain("Task state: continuation");
		expect(selectorPrompt).toContain("Tools enabled: yes");
		expect(selectorPrompt).toContain("route-provider/candidate-a");
		expect(selectorPrompt).not.toContain("old assistant response");
	});

	test("does not force selector temperature", async () => {
		writeConfig();
		let selectorOptions: StreamOptions | undefined;
		faux.setResponses([
			(_context, options) => {
				selectorOptions = options;
				return fauxAssistantMessage(JSON.stringify({ model: "route-provider/candidate-a", reason: "small task" }));
			},
		]);
		const { registry, router } = createRegistryAndRouter();

		await router.resolve({
			requestedModel: getRouterModel(registry),
			context: createRouterContext(),
		});

		expect(selectorOptions).toBeDefined();
		expect(selectorOptions?.temperature).toBeUndefined();
		expect(selectorOptions?.maxTokens).toBe(256);
	});

	test("uses fallback for invalid selector JSON", async () => {
		writeConfig({ fallback: "route-provider/candidate-b" });
		faux.setResponses([fauxAssistantMessage("not json")]);
		const { registry, router } = createRegistryAndRouter();

		const decision = await router.resolve({
			requestedModel: getRouterModel(registry),
			context: createRouterContext(),
		});

		expect(decision?.model.id).toBe("candidate-b");
		expect(decision?.fallbackUsed).toBe(true);
	});

	test("uses fallback when the selector returns a non-candidate", async () => {
		writeConfig({ fallback: "route-provider/candidate-a" });
		faux.setResponses([
			fauxAssistantMessage(JSON.stringify({ model: "route-provider/not-a-candidate", reason: "bad id" })),
		]);
		const { registry, router } = createRegistryAndRouter();

		const decision = await router.resolve({
			requestedModel: getRouterModel(registry),
			context: createRouterContext(),
		});

		expect(decision?.model.id).toBe("candidate-a");
		expect(decision?.fallbackUsed).toBe(true);
	});

	test("uses fallback when the selector call fails", async () => {
		writeConfig({ fallback: "route-provider/candidate-b" });
		faux.setResponses([fauxAssistantMessage("", { stopReason: "error", errorMessage: "selector failed" })]);
		const { registry, router } = createRegistryAndRouter();

		const decision = await router.resolve({
			requestedModel: getRouterModel(registry),
			context: createRouterContext(),
		});

		expect(decision?.model.id).toBe("candidate-b");
		expect(decision?.fallbackUsed).toBe(true);
	});

	test("rejects selector models that resolve to routers", () => {
		writeConfig({ selectorModel: "router/upstart-router" });
		const registry = ModelRegistry.create(authStorage, modelsJsonPath);

		expect(registry.getError()).toContain("selectorModel must resolve to a concrete model");
	});

	test("fails before streaming when no candidates are eligible", async () => {
		const missingEnv = "PI_TEST_MISSING_ROUTER_CANDIDATE_AUTH";
		const originalEnv = process.env[missingEnv];
		const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
		delete process.env[missingEnv];

		try {
			writeConfig({
				providers: {
					"selector-provider": {
						baseUrl: "https://selector.test/v1",
						apiKey: "test-key",
						api: faux.api,
						models: [{ id: "selector" }],
					},
					"candidate-provider": {
						baseUrl: "https://candidate.test/v1",
						apiKey: `$${missingEnv}`,
						api: faux.api,
						models: [{ id: "candidate-a" }],
					},
				},
				selectorModel: "selector-provider/selector",
				candidates: ["candidate-provider/candidate-a"],
			});
			const { registry, router } = createRegistryAndRouter();

			await expect(
				router.resolve({
					requestedModel: getRouterModel(registry),
					context: createRouterContext(),
				}),
			).rejects.toThrow("no eligible candidates");
			expect(faux.state.callCount).toBe(0);
			expect(warnSpy).toHaveBeenCalledWith(
				expect.stringContaining('Warning: No models match pattern "candidate-provider/candidate-a"'),
			);
		} finally {
			warnSpy.mockRestore();
			if (originalEnv === undefined) {
				delete process.env[missingEnv];
			} else {
				process.env[missingEnv] = originalEnv;
			}
		}
	});
});
