def compute_similarity_matrix(df, feature_columns):
    features = df[feature_columns]
    return features.corr().abs()


def score_axis_order(similarity_matrix, order):
    """
    Compute the average similarity between neighboring axes.
    """

    similarities = []

    for left, right in zip(order[:-1], order[1:]):
        similarities.append(similarity_matrix.loc[left, right])

    return sum(similarities) / len(similarities)


def greedy_axis_order_from_start(similarity_matrix, start_axis, mode="SIM"):
    """
    Create one greedy axis order from a given start axis.
    """

    remaining_axes = list(similarity_matrix.columns)

    current_axis = start_axis
    remaining_axes.remove(current_axis)

    order = [current_axis]

    while remaining_axes:

        if mode == "SIM":
            next_axis = max(
                remaining_axes,
                key=lambda axis: similarity_matrix.loc[current_axis, axis]
            )

        elif mode == "DIS":
            next_axis = min(
                remaining_axes,
                key=lambda axis: similarity_matrix.loc[current_axis, axis]
            )

        else:
            raise ValueError("mode must be 'SIM' or 'DIS'")

        order.append(next_axis)
        remaining_axes.remove(next_axis)
        current_axis = next_axis

    return order


def greedy_axis_order(similarity_matrix, mode="SIM"):
    """
    Try every possible start axis and keep the best greedy result.

    SIM:
        keep the order with the highest average neighboring similarity

    DIS:
        keep the order with the lowest average neighboring similarity
    """

    candidate_orders = []

    for start_axis in similarity_matrix.columns:
        order = greedy_axis_order_from_start(
            similarity_matrix,
            start_axis,
            mode=mode
        )

        score = score_axis_order(similarity_matrix, order)

        candidate_orders.append(
            (order, score)
        )

    if mode == "SIM":
        best_order, best_score = max(
            candidate_orders,
            key=lambda item: item[1]
        )

    elif mode == "DIS":
        best_order, best_score = min(
            candidate_orders,
            key=lambda item: item[1]
        )

    else:
        raise ValueError("mode must be 'SIM' or 'DIS'")

    print(f"\nBest {mode} score: {best_score:.3f}")

    return best_order


def print_axis_order_debug(similarity_matrix, order, name):
    print(f"\n{name} axis order:")
    print(order)

    print(f"\n{name} neighboring similarities:")

    values = []

    for left, right in zip(order[:-1], order[1:]):
        similarity = similarity_matrix.loc[left, right]
        values.append(similarity)
        print(f"{left:25s} -> {right:25s}: {similarity:.3f}")

    average_similarity = sum(values) / len(values)

    print(f"\n{name} average neighboring similarity: {average_similarity:.3f}")

    return average_similarity


if __name__ == "__main__":
    from preprocessing import (
        load_and_preprocess,
        load_and_preprocess_cars
    )

    # Wine Dataset Debug
    print("WINE DATASET")

    wine_df = load_and_preprocess()

    wine_features = wine_df.drop(
        columns=["class"]
    ).columns

    wine_similarity_matrix = compute_similarity_matrix(
        wine_df,
        wine_features
    )

    wine_sim_order = greedy_axis_order(
        wine_similarity_matrix,
        mode="SIM"
    )

    wine_dis_order = greedy_axis_order(
        wine_similarity_matrix,
        mode="DIS"
    )


    wine_sim_score = print_axis_order_debug(
        wine_similarity_matrix,
        wine_sim_order,
        "Wine SIM"
    )

    wine_dis_score = print_axis_order_debug(
        wine_similarity_matrix,
        wine_dis_order,
        "Wine DIS"
    )


    print("\nWine Summary:")
    print(f"SIM score: {wine_sim_score:.3f}")
    print(f"DIS score: {wine_dis_score:.3f}")


    # Cars Dataset Debug
    print("CARS DATASET")
    cars_df, cars_original = load_and_preprocess_cars()

    cars_features = cars_df.drop(
        columns=[
            "origin",
            "model_year"
        ]
    ).columns


    cars_similarity_matrix = compute_similarity_matrix(
        cars_df,
        cars_features
    )


    cars_sim_order = greedy_axis_order(
        cars_similarity_matrix,
        mode="SIM"
    )


    cars_dis_order = greedy_axis_order(
        cars_similarity_matrix,
        mode="DIS"
    )


    cars_sim_score = print_axis_order_debug(
        cars_similarity_matrix,
        cars_sim_order,
        "Cars SIM"
    )


    cars_dis_score = print_axis_order_debug(
        cars_similarity_matrix,
        cars_dis_order,
        "Cars DIS"
    )


    print("\nCars Summary:")
    print(f"SIM score: {cars_sim_score:.3f}")
    print(f"DIS score: {cars_dis_score:.3f}")