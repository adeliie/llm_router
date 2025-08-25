import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import os

def load_cached_results():
    """Load cached results from cache files"""
    results = {}
    
    # Load small LLM responses cache
    if os.path.exists("caching_results/small_llm_cache_gsm8k.pkl"):
        with open("caching_results/small_llm_cache.pkl", "rb") as f:
            results['small_llm_cache'] = pickle.load(f)
        print(f"✓ Small LLM cache loaded: {len(results['small_llm_cache'])} responses")
    else:
        print("✗ File small_llm_cache.pkl not found")
        results['small_llm_cache'] = {}
    
    # Load score cache
    if os.path.exists("caching_results/score_cache_gsm8k.pkl"):
        with open("caching_results/score_cache_gsm8k.pkl", "rb") as f:
            results['score_cache'] = pickle.load(f)
        print(f"✓ Score cache loaded: {len(results['score_cache'])} scores")
    else:
        print("✗ File score_cache.pkl not found")
        results['score_cache'] = {}
    
    # Load router results
    if os.path.exists("caching_results/router_results_gsm8k.json"):
        with open("caching_results/router_results_gsm8k.json", "r") as f:
            results['router_results'] = json.load(f)
        print(f"✓ Router results loaded: {len(results['router_results'])} thresholds")
    else:
        print("✗ File router_results.json not found")
        results['router_results'] = []
    
    return results

def create_comparison_plot(results):
    """Create the router comparison plot with dynamic zoom"""
    
    if not results['router_results']:
        print("No router results found.")
        return
    
    routers_data = {}
    for router_name, router_results in results['router_results'].items():
        routers_data[router_name] = {
            'pct_big': [r['pct_big'] for r in router_results],
            'avg_score': [r['avg_score'] for r in router_results],
            'tau': [r['tau'] for r in router_results]
        }

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    # Collect all values to determine zoom
    all_pct = []
    all_scores = []
    for name, data in routers_data.items():
        if name != "Random":
            all_pct.extend(data['pct_big'])
            all_scores.extend(data['avg_score'])
    
    # Add small padding
    pct_pad = (max(all_pct) - min(all_pct)) * 0.05
    score_pad = (max(all_scores) - min(all_scores)) * 0.05

    min_pct, max_pct = min(all_pct) - pct_pad, max(all_pct) + pct_pad
    min_score, max_score = min(all_scores) - score_pad, max(all_scores) + score_pad
    y1 = routers_data['KNN']['avg_score'][-1] if 'KNN' in routers_data else min(all_scores)
    x1, x2 = 0, 100
    y2 = 10
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x_range = np.linspace(min_pct, max_pct, 100)
    y_theoretical = a * x_range + b
    plt.plot(x_range, y_theoretical, '--', color='gray', alpha=0.8,
             label=f'Random (theoretical: y={a:.2f}x+{b:.2f})', linewidth=2)
    
    for i, (router_name, data) in enumerate(routers_data.items()):
        color = colors[i % len(colors)]
        plt.plot(data['pct_big'], data['avg_score'], 'o-', color=color, linewidth=2,
                 markersize=6, label=f'{router_name} Router')

    # Reference Big/Small LLM
    plt.axhline(10.0, linestyle='--', color='red', alpha=0.7, label='Big LLM (Perfect score)')
    if results['score_cache']:
        avg_small_score = np.mean(list(results['score_cache'].values()))
        plt.axhline(avg_small_score, linestyle=':', color='gray', alpha=0.7,
                    label=f'Small LLM (average={avg_small_score:.2f})')

    plt.xlabel('% of calls to Big LLM', fontsize=14)
    plt.ylabel('Average score (0-10)', fontsize=14)
    plt.title('Comparison of LLM Routers (Zoomed)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')

    # Zoomed limits
    plt.xlim(min_pct, max_pct)
    plt.ylim(min_score, max_score)
    plt.minorticks_on()
    plt.grid(True, which='minor', alpha=0.2)
    
    return plt

def save_plot(plt, filename="evaluation_and_visualization/router_comparison.png", dpi=300):
    """Save the figure in high quality"""
    try:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Figure saved: {filename}")
        
    except Exception as e:
        print(f"✗ Error while saving: {e}")

def main():
    """Main function"""
    print("=== LLM ROUTERS RESULTS VISUALIZER ===\n")
    
    # Load results
    results = load_cached_results()
    
    if not results['router_results']:
        print("\n❌ No results found. Please run evaluation.py first")
        return

    # Create figure
    print("\n📊 Creating comparison figure...")
    plt = create_comparison_plot(results)
    
    if plt:
        # Save figure
        print("\n💾 Saving figure...")
        save_plot(plt, "evaluation_and_visualization/router_comparison_gsm8k.png")
        
        # Show figure
        print("\n🖼️  Displaying figure...")
        plt.tight_layout()
        plt.show()
        
        print("\n✅ Visualization completed successfully!")
    else:
        print("\n❌ Could not create figure")

if __name__ == "__main__":
    main()
