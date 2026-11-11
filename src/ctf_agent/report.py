import json
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def generate_dashboard(runs_dir: Path, output_html: Path):
    """Generates a high-fidelity analytics dashboard for CTF runs."""
    # Find all summary files recursively
    summary_files = sorted(runs_dir.rglob("summary.json"))
    if not summary_files:
        print("No results found to aggregate.")
        return

    # Group runs by challenge
    by_chal = defaultdict(list)
    for f in summary_files:
        data = json.loads(f.read_text())
        # Use challenge_id as key, store the trial and result
        by_chal[data["challenge_id"]].append(data)

    # Calculate metrics per challenge
    chal_metrics = {}
    total_trials = 0
    total_solved = 0
    
    for cid, runs in by_chal.items():
        n = len(runs)
        total_trials += n
        outcomes = [int(r.get("verified", False)) for r in runs]
        c = sum(outcomes)
        total_solved += c
        
        # Resource aggregation
        in_toks = [r.get("input_tokens") for r in runs if r.get("input_tokens") is not None]
        out_toks = [r.get("output_tokens") for r in runs if r.get("output_tokens") is not None]
        steps = [r.get("steps") for r in runs if r.get("steps") is not None]
        elapsed = [r.get("elapsed_s") for r in runs if r.get("elapsed_s") is not None]
        
        chal_metrics[cid] = {
            "n": n,
            "c": c,
            "rate": (c/n*100) if n > 0 else 0,
            "pass_at_k": {k: pass_at_k(n, c, k) for k in range(1, n + 1)},
            "avg_in": mean(in_toks) if in_toks else 0,
            "avg_out": mean(out_toks) if out_toks else 0,
            "avg_steps": mean(steps) if steps else 0,
            "avg_time": mean(elapsed) if elapsed else 0,
        }

    overall_rate = (total_solved / total_trials * 100) if total_trials > 0 else 0

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CTF Agent Analytics Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
                   background: #1a1b26; color: #a9b1d6; margin: 0; padding: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #7aa2f7; text-align: center; font-size: 2.5em; margin-bottom: 40px; }}
            
            .top-metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }}
            .metric-card {{ background: #24283b; padding: 20px; border-radius: 12px; border: 1px solid #414868; text-align: center; }}
            .metric-card .val {{ font-size: 2.2em; font-weight: bold; display: block; color: #9ece6a; }}
            .metric-card .lbl {{ font-size: 0.9em; color: #565f89; text-transform: uppercase; letter-spacing: 1px; }}
            
            table {{ width: 100%; border-collapse: collapse; background: #24283b; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
            th {{ background: #414868; color: #c0caf5; padding: 15px; text-align: left; font-weight: bold; }}
            td {{ padding: 12px 15px; border-bottom: 1px solid #414868; }}
            tr:hover {{ background: #2f334d; }}
            
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
            .badge-solved {{ background: #454f43; color: #9ece6a; }}
            .badge-failed {{ background: #453d43; color: #f7768e; }}
            
            .progress-bar {{ background: #1a1b26; height: 8px; border-radius: 4px; width: 100px; display: inline-block; vertical-align: middle; margin-right: 10px; }}
            .progress-fill {{ background: #9ece6a; height: 100%; border-radius: 4px; }}
            
            .section-title {{ color: #bb9af7; margin: 40px 0 20px 0; font-size: 1.5em; border-left: 4px solid #bb9af7; padding-left: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Agent Performance Dashboard</h1>
            
            <div class="top-metrics">
                <div class="metric-card"><span class="val">{total_trials}</span><span class="lbl">Total Trials</span></div>
                <div class="metric-card"><span class="val">{total_solved}</span><span class="lbl">Total Solved</span></div>
                <div class="metric-card"><span class="val">{overall_rate:.1f}%</span><span class="lbl">Global Solve Rate</span></div>
                <div class="metric-card"><span class="val">{len(chal_metrics)}</span><span class="lbl">Challenges</span></div>
            </div>

            <div class="section-title">Detailed Breakdown</div>
            <table>
                <thead>
                    <tr>
                        <th>Challenge ID</th>
                        <th>Status</th>
                        <th>Pass@1</th>
                        <th>Pass@3</th>
                        <th>Avg Steps</th>
                        <th>Avg Tokens (In/Out)</th>
                        <th>Avg Time</th>
                    </tr>
                </thead>
                <tbody>
    """
    for cid, m in sorted(chal_metrics.items()):
        is_solved = m["c"] > 0
        status_cls = "badge-solved" if is_solved else "badge-failed"
        status_text = "SOLVED" if is_solved else "FAILED"
        
        p1 = m["pass_at_k"].get(1, 0) * 100
        p3 = m["pass_at_k"].get(3, 0) * 100
        
        html += f"""
                <tr>
                    <td style="font-weight:bold; color:#7aa2f7;">{cid}</td>
                    <td><span class="badge {status_cls}">{status_text}</span></td>
                    <td>
                        <div class="progress-bar"><div class="progress-fill" style="width: {p1}%"></div></div>
                        {p1:.1f}%
                    </td>
                    <td>
                        <div class="progress-bar"><div class="progress-fill" style="width: {p3}%"></div></div>
                        {p3:.1f}%
                    </td>
                    <td>{m["avg_steps"]:.1f}</td>
                    <td>{m["avg_in"]:,.0f} / {m["avg_out"]:,.0f}</td>
                    <td>{m["avg_time"]:.1f}s</td>
                </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    output_html.write_text(html)
    print(f"Dashboard generated: {output_html}")
