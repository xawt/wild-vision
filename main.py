import torch


def main():
    print("Hello from wild-vision!\n")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    print("")


if __name__ == "__main__":
    main()
