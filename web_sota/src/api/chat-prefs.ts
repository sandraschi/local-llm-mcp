const KEY = "llm-mcp-chat-prefs";

export interface ChatPrefs {
  personalityId: string;
  promptRefinement: boolean;
}

export function getChatPrefs(): ChatPrefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { personalityId: "neutral", promptRefinement: false };
    const d = JSON.parse(raw) as ChatPrefs;
    return {
      personalityId: typeof d.personalityId === "string" ? d.personalityId : "neutral",
      promptRefinement: Boolean(d.promptRefinement),
    };
  } catch {
    return { personalityId: "neutral", promptRefinement: false };
  }
}

export function setChatPrefs(prefs: Partial<ChatPrefs>): void {
  const current = getChatPrefs();
  const next = { ...current, ...prefs };
  localStorage.setItem(KEY, JSON.stringify(next));
}
