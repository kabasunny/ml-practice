from itertools import combinations


def generate_model_combinations(model_types):
    model_combinations = []
    for r in range(2, len(model_types) + 1):  # 組み合わせのサイズを2以上に設定
        model_combinations.extend(combinations(model_types, r))

    print("\nモデルの組み合わせ:")
    for combination in model_combinations:
        print(combination)

    return model_combinations