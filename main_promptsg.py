import os
import argparse

from main import run_reid
from modules.promptsg import PromptSGDescriptor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run vehicle Re-ID with the PromptSG descriptor')
    parser.add_argument('--model_path', type=str, default='checkpoints_promptsg/best_promptsg_model.pth')
    parser.add_argument('--prompt', type=str, default='simplified', choices=['simplified', 'composed'],
                        help='simplified: fixed "a photo of a vehicle." (fast); '
                             'composed: per-crop pseudo-token prompt (runs the text encoder, ~2x slower)')
    args = parser.parse_args()

    video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "videos")
    video1 = os.path.join(video_dir, "Cam_1.mp4")
    video2 = os.path.join(video_dir, "Cam_2.mp4")
    output_dir = "results_promptsg"
    os.makedirs(output_dir, exist_ok=True)

    run_reid(
        video1,
        video2,
        save_dir=output_dir,
        # PromptSG embeddings are 1024-d, so they need their own collection apart from the 512-d OSNet ones
        db_path="./chroma_vehicle_reid_promptsg",
        reset_database=True,
        descriptor=PromptSGDescriptor(model_path=args.model_path, inference_prompt=args.prompt),
    )
