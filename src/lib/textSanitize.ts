export function stripModelThinking(text: string): string {
  return text
    .replace(/```[a-zA-Z]*\s*/g, "")
    .replace(/```/g, "")
    .replace(/<?think>\s*[\s\S]*?<\/think>/gi, "")
    .replace(/<thinking>\s*[\s\S]*?<\/thinking>/gi, "")
    .replace(/\bthinking\s+[\s\S]*?\bresponse\b/gi, "")
    .trim();
}
