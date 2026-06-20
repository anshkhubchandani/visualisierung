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
    from src.preprocessing import load_and_preprocess

    df = load_and_preprocess()
    feature_columns = df.drop(columns=["class"]).columns

    similarity_matrix = compute_similarity_matrix(df, feature_columns)

    sim_order = greedy_axis_order(similarity_matrix, mode="SIM")
    dis_order = greedy_axis_order(similarity_matrix, mode="DIS")

    sim_score = print_axis_order_debug(
        similarity_matrix,
        sim_order,
        "SIM"
    )

    dis_score = print_axis_order_debug(
        similarity_matrix,
        dis_order,
        "DIS"
    )

    print("\nSummary:")
    print(f"SIM score: {sim_score:.3f}")
    print(f"DIS score: {dis_score:.3f}")