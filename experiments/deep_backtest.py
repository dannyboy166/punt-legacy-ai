"""
Deep backtest: same data + prompt as live predictor, but requests
per-runner analysis and a detailed race summary.

Usage:
    python experiments/deep_backtest.py "Sandown-Hillside" 5 "28-Jan-2026"
"""
import sys
sys.path.insert(0, '/Users/danielsamus/punt-legacy-ai')

from experiments.backtest import get_backtest_prompt, SYSTEM_PROMPT
import anthropic
import json
import re

# Same system prompt as live, but override the output format only
DEEP_OUTPUT = """

## Output

```json
{
  "runners": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "verdict": "2-4 sentences: speed ratings analysis, form assessment, strengths, weaknesses, and whether odds are fair"
    }
  ],
  "contenders": [
    {
      "horse": "Horse Name",
      "tab_no": number,
      "odds": number,
      "place_odds": number,
      "tag": "The one to beat" | "Each-way chance" | "Value bet",
      "analysis": "1-2 sentences referencing RACE form",
      "tipsheet_pick": true | false
    }
  ],
  "race_summary": "Detailed 3-5 sentence summary: pace scenario, key factors, why you picked (or didn't pick) contenders, and overall confidence level"
}
```"""

# Build deep system prompt: keep everything before "## Output", replace output section
def build_deep_system_prompt():
    # Split at "## Output" and replace just that section
    parts = SYSTEM_PROMPT.split("## Output")
    if len(parts) == 2:
        return parts[0] + DEEP_OUTPUT + "\n\n" + parts[1].split("```")[2] if "```" in parts[1] else parts[0] + DEEP_OUTPUT
    return SYSTEM_PROMPT + DEEP_OUTPUT

# Rebuild: keep everything before ## Output, add deep output, keep tipsheet_pick section after
def get_deep_system_prompt():
    before_output = SYSTEM_PROMPT.split("## Output")[0]
    # Get the tipsheet_pick section (after the closing ```)
    after_json = SYSTEM_PROMPT.split("```\n\n**tipsheet_pick")
    tipsheet_section = "**tipsheet_pick" + after_json[1] if len(after_json) > 1 else ""
    return before_output + DEEP_OUTPUT + "\n\n" + tipsheet_section


def run_deep_prediction(prompt_text):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        temperature=0.2,
        system=get_deep_system_prompt(),
        messages=[{"role": "user", "content": f"Analyze this race and pick your contenders (0-3). Provide a verdict for EVERY runner.\n\n{prompt_text}\n\nRespond with JSON only."}]
    )
    return response.content[0].text


if __name__ == "__main__":
    track = sys.argv[1]
    race_num = int(sys.argv[2])
    date = sys.argv[3]

    print(f"\n{'='*60}")
    print(f"  DEEP BACKTEST: {track} R{race_num} - {date}")
    print(f"{'='*60}")

    result = get_backtest_prompt(track, race_num, date)

    if result[0] is None:
        print(f"Error: {result[1]}")
        sys.exit(1)

    prompt_text, runners, race_info = result

    print(f"\nDistance: {race_info['distance']}m | Condition: {race_info['condition']}")
    print(f"Class: {race_info['name']}")
    print(f"Form: {race_info['with_form']}/{race_info['total']} have race runs")
    print(f"\nFIELD:")
    for r in runners:
        print(f"  {r['tab']}. {r['name']} - SP ${r['sp']:.2f}")

    # Print the raw form data (same data Claude sees)
    print(f"\n{'='*60}")
    print("  FORM DATA (fed to Claude)")
    print(f"{'='*60}")
    print(prompt_text)

    print(f"\n{'='*60}")
    print("  RUNNER ANALYSIS")
    print(f"{'='*60}")

    raw = run_deep_prediction(prompt_text)

    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        data = json.loads(json_match.group()) if json_match else {}

        # Per-runner verdicts
        for rv in data.get('runners', []):
            print(f"\n  {rv.get('tab_no', '?')}. {rv['horse']} (${rv.get('odds', '?')})")
            print(f"     {rv['verdict']}")

        # Contenders
        contenders = data.get('contenders', [])
        print(f"\n{'='*60}")
        print("  CONTENDERS")
        print(f"{'='*60}")

        if not contenders:
            print(f"\n  NO CONTENDERS")
        else:
            for c in contenders:
                tipsheet = " ⭐ TIPSHEET" if c.get('tipsheet_pick') else ""
                print(f"\n  {c['horse']} (#{c['tab_no']}){tipsheet}")
                print(f"    ${c['odds']:.2f} win / ${c.get('place_odds', 0):.2f} place")
                print(f'    "{c["tag"]}"')
                print(f"    {c['analysis']}")

        # Race summary
        print(f"\n{'='*60}")
        print("  RACE SUMMARY")
        print(f"{'='*60}")
        print(f"\n  {data.get('race_summary', 'No summary')}")

    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw: {raw[:500]}")

    print(f"\n{'='*60}")
