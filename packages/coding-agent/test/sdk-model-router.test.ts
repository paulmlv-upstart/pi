import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type AssistantMessage, fauxAssistantMessage, registerFauxProvider } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import type { AgentSessionEvent } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { ModelRegistry } from "../src/core/model-registry.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";

describe("createAgentSession model router", () => {
	let tempDir: string;
	let cwd: string;
	let agentDir: string;
	let modelsJsonPath: string;
	let faux: ReturnType<typeof registerFauxProvider>;

	beforeEach(() => {
		tempDir = mkdtempSync(join(tmpdir(), "pi-sdk-model-router-"));
		cwd = join(tempDir, "project");
		agentDir = join(tempDir, "agent");
		mkdirSync(cwd, { recursive: true });
		mkdirSync(agentDir, { recursive: true });
		modelsJsonPath = join(agentDir, "models.json");
		faux = registerFauxProvider({
			provider: "target-provider",
			models: [{ id: "selector" }, { id: "candidate-a" }, { id: "candidate-b" }],
		});
	});

	afterEach(() => {
		faux.unregister();
		rmSync(tempDir, { recursive: true, force: true });
	});

	function writeRouterConfig(): void {
		writeFileSync(
			modelsJsonPath,
			JSON.stringify({
				providers: {
					"selector-provider": {
						baseUrl: "https://selector.test/v1",
						apiKey: "selector-key",
						api: faux.api,
						models: [{ id: "selector" }],
					},
					"target-provider": {
						baseUrl: "https://target.test/v1",
						apiKey: "target-key",
						api: faux.api,
						models: [{ id: "candidate-a" }, { id: "candidate-b" }],
					},
				},
				routers: {
					"upstart-router": {
						selectorModel: "selector-provider/selector",
						candidates: ["target-provider/candidate-a", "target-provider/candidate-b"],
						fallback: "target-provider/candidate-a",
					},
				},
			}),
		);
	}

	test("selects with the selector model and streams the concrete target", async () => {
		writeRouterConfig();
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, modelsJsonPath);
		const routerModel = modelRegistry.find("router", "upstart-router");
		if (!routerModel) throw new Error("router model not found");

		const settingsManager = SettingsManager.inMemory({});
		const sessionManager = SessionManager.inMemory(cwd);
		const calls: Array<{ provider: string; model: string; apiKey: string | undefined }> = [];

		faux.setResponses([
			(_context, options, _state, model) => {
				calls.push({ provider: model.provider, model: model.id, apiKey: options?.apiKey });
				return fauxAssistantMessage(
					JSON.stringify({ model: "target-provider/candidate-b", reason: "needs stronger coding model" }),
				);
			},
			(_context, options, _state, model) => {
				calls.push({ provider: model.provider, model: model.id, apiKey: options?.apiKey });
				return fauxAssistantMessage("target ok");
			},
		]);

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model: routerModel,
			authStorage,
			modelRegistry,
			settingsManager,
			sessionManager,
		});
		const events: AgentSessionEvent[] = [];
		session.subscribe((event) => {
			events.push(event);
		});

		try {
			await session.prompt("fix the router test");
		} finally {
			session.dispose();
		}

		expect(calls).toEqual([
			{ provider: "selector-provider", model: "selector", apiKey: "selector-key" },
			{ provider: "target-provider", model: "candidate-b", apiKey: "target-key" },
		]);

		const statusEvents = events.filter((event) => event.type === "status");
		expect(statusEvents).toEqual([
			{ type: "status", message: "Using target-provider/candidate-b via upstart-router for this task." },
		]);

		const lastMessage = session.state.messages[session.state.messages.length - 1];
		expect(lastMessage?.role).toBe("assistant");
		const assistantMessage = lastMessage as AssistantMessage;
		expect(assistantMessage.provider).toBe("target-provider");
		expect(assistantMessage.model).toBe("candidate-b");
		expect(assistantMessage.content).toEqual([{ type: "text", text: "target ok" }]);
		expect(assistantMessage.diagnostics).toContainEqual(
			expect.objectContaining({
				type: "model_router",
				details: expect.objectContaining({
					router: "upstart-router",
					selected: "target-provider/candidate-b",
					selectorModel: "selector-provider/selector",
					reason: "needs stronger coding model",
					eligibleCandidates: 2,
					fallbackUsed: false,
				}),
			}),
		);

		const modelChanges = sessionManager.getBranch().filter((entry) => entry.type === "model_change");
		expect(modelChanges).toHaveLength(1);
		expect(modelChanges[0]).toMatchObject({
			type: "model_change",
			provider: "router",
			modelId: "upstart-router",
		});
	});
});
