import numpy as np
import statistics

import service.train_service as ts


def stochastic_dominance(galaxy_images, indexed_labels):
    model_names = ["valid/L_CUSTOM_2_3_64_90240_10ep_96.37acc.h5", "L_CUSTOM_4_1_64_112920_15ep_acc93.h5", "L_RESNET_1_0_64_169176_10ep.h5", "L_MOBILE_1_1_64_169176_20ep.h5", "L_EFFICIENT_1_0_64_60160_10ep.h5", "L_DENSENET_1_2_64_169176_20ep.h5"]
    number_of_classes = len(set(indexed_labels))
    final_results = []
    final_scores = []
    for model_name in model_names:
        results, scores = ts.evaluate(galaxy_images, indexed_labels, model_name, manual=True)
        final_results.append(results)
        final_scores.append(scores)

    eval_results = []
    metrics = []
    # network_score_means = []
    # network_score_sds = []
    for network_id in range(len(model_names)):
        eval_results.append([])
        score_sum = 0
        for image_id in range(len(galaxy_images)):
            expected = final_results[network_id][image_id][0]
            actual = final_results[network_id][image_id][1]
            score = final_scores[network_id][image_id].max()
            score_sum += score
            eval_results[network_id].append({"expected": expected, "actual": actual, "confidence": score})
        network_scores = np.array([d['confidence'] for d in eval_results[network_id]])
        metrics.append({'mean': network_scores.mean(), 'median': np.median(network_scores), 'sd': np.std(network_scores), 'var': np.var(network_scores), 'iqr': np.percentile(network_scores, [25, 50, 75]), 'cv': np.std(network_scores) / np.mean(network_scores) * 100})
        # network_score_means.append(np.mean(network_scores))
        # network_score_means.append(np.std(network_scores))

    network_winners = []
    for image_id in range(len(galaxy_images)):
        print("Image " + str(image_id) + " results:")
        max_confidence = 0
        better_network = -1
        for network_id in range(len(model_names)):
            network_confidence = eval_results[network_id][image_id]['confidence']
            print("network " + str(network_id) + " confidence: " + str(network_confidence))
            if network_confidence > max_confidence and eval_results[network_id][image_id]['actual'] == eval_results[network_id][image_id]['expected']:
                max_confidence = network_confidence
                better_network = network_id
        network_winners.append(better_network)
        print("Winner network: " + str(better_network))

    network_winners_count = []
    for network_id in range(len(model_names)):
        network_winners_count.append(network_winners.count(network_id))

    print("Network winning counts: ", network_winners_count)
    print("Metrics: ", metrics)
    # print("Averages: ", network_score_means)
    # print("Standard deviations: ", network_score_sds)
    print("Stochastic dominant network: " + str(np.argmax(network_winners_count, axis=-1)))
