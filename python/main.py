"""
SortBot Vision System
=====================
Loads a random batch of CIFAR-10 images at every launch,
runs them through an OpenCV preprocessing + classification pipeline,
then feeds them into a pygame conveyor belt game.

Usage:
    python main.py                  # default: 60 images, color classifier
    python main.py --n 80 --clf ml  # 80 images, ML classifier
    python main.py --help
"""

import argparse
from core.dataset import load_random_batch
from core.pipeline import VisionPipeline
from game.conveyor import ConveyorGame


def parse_args():
    p = argparse.ArgumentParser(description="SortBot Vision Sorting Game")
    p.add_argument("--n",    type=int, default=60,      help="Images per run (default 60)")
    p.add_argument("--clf",  type=str, default="color",
                   choices=["color", "hog", "ml"],      help="Classifier mode")
    p.add_argument("--noise",type=int, default=0,
                   choices=[0,1,2,3,4],                 help="Simulated noise level 0-4")
    p.add_argument("--speed",type=int, default=2,
                   choices=[1,2,3,4,5],                 help="Belt speed 1-5")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  SortBot — Vision Sorting System")
    print(f"  Classifier : {args.clf.upper()}")
    print(f"  Batch size : {args.n} images")
    print(f"  Noise level: {args.noise}")
    print(f"  Belt speed : {args.speed}")
    print(f"{'='*50}\n")

    print("[1/3] Loading CIFAR-10 batch...")
    images, labels, class_names = load_random_batch(n=args.n)
    print(f"      Loaded {len(images)} images across {len(set(labels))} classes\n")

    print("[2/3] Initialising vision pipeline...")
    pipeline = VisionPipeline(classifier=args.clf, noise_level=args.noise)
    print(f"      Pipeline ready: preprocess → segment → {args.clf} classifier\n")

    print("[3/3] Launching conveyor game — close window to finish\n")
    game = ConveyorGame(
        images=images,
        labels=labels,
        class_names=class_names,
        pipeline=pipeline,
        belt_speed=args.speed,
    )
    results = game.run()

    print("\n" + "="*50)
    print("  SESSION RESULTS")
    print("="*50)
    print(f"  Total sorted : {results['total']}")
    print(f"  Correct      : {results['correct']}")
    acc = results['correct'] / max(results['total'], 1) * 100
    print(f"  Accuracy     : {acc:.1f}%")
    print(f"  Score        : {results['score']}")
    print(f"  Best streak  : {results['best_streak']}")
    print(f"\n  Per-class accuracy:")
    for cls, (c, t) in results['per_class'].items():
        bar = "#" * int(c/max(t,1)*20)
        print(f"    {cls:<12} {c:>3}/{t:<3}  [{bar:<20}]  {c/max(t,1)*100:.0f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
