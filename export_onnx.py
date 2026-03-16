import torch
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
import argparse
import os
from onnx_wrapper import InferenceWrapper


# ------------------------ FIX START --------------------------
# Manually inject 'used_layers' if missing
if "used_layers" not in cfg.BACKBONE.KWARGS:
    print("⚠️ 'used_layers' not found in cfg.BACKBONE.KWARGS, injecting manually.")
    cfg.BACKBONE.KWARGS['used_layers'] = [2, 3, 4]
else:
    print("✅ used_layers found in config:", cfg.BACKBONE.KWARGS['used_layers'])
# ------------------------ FIX END ----------------------------

# build and load model
model = ModelBuilder()
model.load_state_dict(torch.load(args.snapshot, map_location='cpu'))
model.eval()

wrapper = InferenceWrapper(model)

dummy_z = torch.randn(1, 3, 127, 127)
dummy_x = torch.randn(1, 3, 255, 255)

torch.onnx.export(wrapper,
                  (dummy_z, dummy_x),
                  args.output,
                  input_names=["template", "search"],
                  output_names=["cls", "loc"],
                  opset_version=11)
                  
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="siamrpn.onnx", help="ONNX output file name")
    return parser.parse_args()

def main():
    args = get_args()

    # Load config
    cfg.merge_from_file(args.config)

    # Build model
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot, map_location='cpu'))
    model.eval()

    # Dummy inputs for template and search image
    dummy_z = torch.randn(1, 3, 127, 127)  # Template image
    dummy_x = torch.randn(1, 3, 255, 255)  # Search image

    # Export to ONNX
    torch.onnx.export(model, (dummy_z, dummy_x), args.output,
                      verbose=True, opset_version=11,
                      input_names=["z", "x"],
                      output_names=["cls", "loc"])
    
    print(f"[INFO] Exported model to {args.output}")

if __name__ == "__main__":
    main()

