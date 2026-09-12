from .policy import MoveFeatures, WormChessPolicy


def main() -> None:
    policy = WormChessPolicy()
    opening = [
        ("e7e5", MoveFeatures(.57, 1, .57, .71, .2, center=.8, development=.5)),
        ("c7c5", MoveFeatures(.29, 1, .29, .71, .2, center=.5, development=.5)),
        ("g8f6", MoveFeatures(.86, 1, .71, .71, .6, center=.7, development=1)),
    ]
    move, scores = policy.choose(opening)
    print("selected:", move)
    for row in scores:
        print(f"  {row['move']}: {row['score']:+.5f}")


if __name__ == "__main__":
    main()
