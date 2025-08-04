import os
import argparse
import numpy as np
import tensorflow as tf
from model import SeqLabelVAE
from datetime import datetime
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=300)
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pitch_x_axis', type=int, default=68)
    parser.add_argument('--pitch_y_axis', type=int, default=105)
    parser.add_argument('--channels', type=int, default=9)
    parser.add_argument('--timesteps', type=int, default=32)
    parser.add_argument('--no_classes', type=int, default=3)
    parser.add_argument('--test_sequences', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    parser.add_argument('--vae_weights_file', type=str, required=True)
    parser.add_argument('--classifier_weights_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_plots', action='store_true', help='Save confusion matrix plot')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--show_plots', action='store_true', default=True, help='Show confusion matrix plot')
    return parser.parse_args()

def load_test_data(sequences_path, labels_path):
    X = np.load(sequences_path)
    y = np.load(labels_path)
    print(f"Loaded test sequences: {X.shape}, test labels: {y.shape}")
    return X, y

def create_classifier_only_model(hidden_dim, timesteps, no_classes, classifier_weights_file):
    latent_input = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    pooled = tf.keras.layers.GlobalAveragePooling1D()(latent_input)
    classifier_output = tf.keras.layers.Dense(no_classes, activation='softmax')(pooled)
    model = tf.keras.Model(latent_input, classifier_output, name="classifier_only")
    model.load_weights(classifier_weights_file)
    return model

def calculate_classification_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    unique_classes = np.unique(y_true)
    metrics = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=unique_classes
    )
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_classes)
    auc_scores = []
    for i in unique_classes:
        try:
            auc = roc_auc_score((y_true == i).astype(int), y_pred_probs[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(np.nan)
    for i, class_idx in enumerate(unique_classes):
        metrics[f'class_{class_idx}'] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'auc': float(auc_scores[i])
        }
    accuracy = np.mean(y_true == y_pred)
    metrics['macro_avg'] = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1)),
        'auc': float(np.nanmean(auc_scores)),
        'accuracy': float(accuracy)
    }
    metrics['confusion_matrix'] = conf_matrix.tolist()
    return metrics, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, show_plot=True):
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    if show_plot:
        plt.show()
    plt.close()
    return cm

def main():
    args = parse_args()
    X_test, y_test = load_test_data(args.test_sequences, args.test_labels)
    print("Loading VAE encoder...")
    vae = SeqLabelVAE(
        feature_dim=args.feature_dim,
        intermediate_dim=args.intermediate_dim,
        hidden_dim=args.hidden_dim,
        pitch_x_axis=args.pitch_x_axis,
        pitch_y_axis=args.pitch_y_axis,
        channels=args.channels,
        timesteps=args.timesteps,
        no_classes=args.no_classes
    )
    _ = vae(X_test[:1])
    vae.load_weights(args.vae_weights_file)
    # Encode test data
    print("Encoding test data...")
    batch_size = args.batch_size
    latent_reps = []
    for i in range(0, X_test.shape[0], batch_size):
        batch = X_test[i:i+batch_size]
        _, _, a_t, _, _, _ = vae.encode(batch)
        latent_reps.append(a_t.numpy())
    latent_reps = np.concatenate(latent_reps, axis=0)
    # Load classifier
    print("Loading classifier head...")
    classifier = create_classifier_only_model(args.hidden_dim, args.timesteps, args.no_classes, args.classifier_weights_file)
    # Predict
    print("Predicting...")
    y_pred_probs = classifier.predict(latent_reps, batch_size=args.batch_size)
    # Metrics
    print("Calculating metrics...")
    metrics, y_pred = calculate_classification_metrics(y_test, y_pred_probs)
    print("\nClassification Metrics:")
    for class_idx in np.unique(y_test):
        m = metrics[f'class_{class_idx}']
        print(f"Class {class_idx}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}, Support={m['support']}")
    macro = metrics['macro_avg']
    print(f"\nMacro Avg: Accuracy={macro['accuracy']:.4f}, Precision={macro['precision']:.4f}, Recall={macro['recall']:.4f}, F1={macro['f1']:.4f}, AUC={macro['auc']:.4f}")
    # Confusion matrix
    print("\nPlotting confusion matrix...")
    class_names = ['Pass', 'Shot', 'Dribble']
    # save_path = None
    # if args.save_plots:
    #     os.makedirs(args.plots_dir, exist_ok=True)
    #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #     save_path = os.path.join(args.plots_dir, f'confusion_matrix_{timestamp}.png')
    save_path = 'seqlabelvae_v2/g1/plots/conf_dutch_smote.png'
    cm = plot_confusion_matrix(y_test, y_pred, class_names=class_names, save_path=save_path, show_plot=args.show_plots)
    print("\nConfusion Matrix:")
    print("True\\Pred\t" + "\t".join(class_names))
    for i, true_class in enumerate(class_names):
        row = f"{true_class}\t"
        for j in range(len(class_names)):
            row += f"{cm[i, j]}\t"
        print(row)

if __name__ == '__main__':
    main() 