import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from color_recommend import recommend_emotion_colors

def plot_weighted_emotion_gradient(emotion_colors):
    colors = [e['color'] for e in emotion_colors]
    weights = np.array([e['score'] for e in emotion_colors])
    weights = weights / weights.sum()
    positions = np.cumsum(weights)
    positions = np.insert(positions, 0, 0)
    color_tuples = [(positions[i], colors[i]) for i in range(len(colors))]
    color_tuples.append((positions[-1], colors[-1]))
    cmap = LinearSegmentedColormap.from_list("weighted_emotion_gradient", color_tuples)
    fig, ax = plt.subplots(figsize=(8, 2))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    mid_positions = (positions[:-1] + positions[1:]) / 2
    for xpos, e in zip(mid_positions, emotion_colors):
        ax.text(xpos * 255, -10, f"{e['label']} ({e['score']:.2f})",
                ha='center', va='bottom', fontsize=10, rotation=45)
    ax.set_axis_off()
    plt.title("color", fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_from_probs(probs):
    """
    probs: torch.tensor or np.array 7D 감정 벡터 (확률)
    """
    # numpy array로 변환
    if not isinstance(probs, np.ndarray):
        probs = probs.cpu().numpy().flatten()
    
    # Top-1 컬러 (가장 영향을 많이 준 감정의 색)
    result_top1 = recommend_emotion_colors(probs, method='top1')
    print(f"추천 메인 색상: {result_top1['color']} ({result_top1['label']} : {result_top1['score']:.2f})")

    # Top-3 팔레트
    result_mix = recommend_emotion_colors(probs, method='mix')
    print("추천 팔레트:")
    for r in result_mix:
        print(f" - {r['color']} ({r['label']} : {r['score']:.2f})")
    
    # 시각화
    plot_weighted_emotion_gradient(result_mix)

