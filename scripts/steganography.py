import cv2
import numpy as np
import os

def display_saliency_map(saliency_map):
    """Display the saliency map without blocking the execution."""
    cv2.imshow('Saliency Map', saliency_map)
    cv2.waitKey(1)  # Non-blocking wait

def compute_custom_saliency(image):
    """Fallback method: Compute saliency using blur-difference."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    saliency_map = cv2.absdiff(gray, blurred)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)

def compute_saliency_map(image):
    """Compute and return the saliency map using OpenCV or fallback method."""
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    if not success or np.sum(saliency_map) == 0:
        print("Saliency map computation failed. Using custom method.")
        saliency_map = compute_custom_saliency(image)
    return (saliency_map * 255).astype(np.uint8)

def calculate_metrics(original_image, encoded_image):
    """Calculate Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((original_image - encoded_image) ** 2)
    if mse == 0:  # Avoid division by zero
        psnr = float('inf')
    else:
        max_pixel_value = 255.0  # Assuming 8-bit image
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return mse, psnr


def encode_message(image, message, saliency_map):
    print("Encoding message...")
    message_bin = ''.join(format(ord(i), '08b') for i in message) + '00000000'  # Null-terminated
    encoded_image = image.copy()
    message_length = len(message_bin)
    idx = 0

    non_zero_saliency_pixels = np.argwhere(saliency_map > 0)
    total_pixels_needed = message_length

    print(f"Total pixels needed for encoding: {total_pixels_needed}")

    total_pixels_processed = 0
    for pixel in non_zero_saliency_pixels:
        i, j = pixel
        if idx < message_length:
            for c in range(3):  # Assuming RGB image
                encoded_image[i, j, c] = (encoded_image[i, j, c] & 0xFE) | int(message_bin[idx])
                idx += 1
                total_pixels_processed += 1
                if idx >= message_length:
                    break
        if total_pixels_processed % 100 == 0:
            encoding_progress = (total_pixels_processed / total_pixels_needed) * 100
            print(f"Encoding progress: {encoding_progress:.2f}%")
        if idx >= message_length:
            break

    print("Message encoded.")
    return encoded_image

def decode_message(encoded_image, saliency_map, message_length):
    print("Decoding message...")

    decoded_message_bin = ''
    non_zero_saliency_pixels = np.argwhere(saliency_map > 0)
    total_bits_decoded = 0

    for pixel in non_zero_saliency_pixels:
        i, j = pixel
        for c in range(3):  # Assuming RGB image
            decoded_message_bin += str(encoded_image[i, j, c] & 1)
            total_bits_decoded += 1
            if total_bits_decoded >= message_length:
                break
        if total_bits_decoded >= message_length:
            break

    # Convert binary to string message
    decoded_message = ''
    for i in range(0, len(decoded_message_bin), 8):
        byte = decoded_message_bin[i:i+8]
        decoded_message += chr(int(byte, 2))

    print("Decoding complete.")
    return decoded_message.strip('\x00')  # Remove null termination

def main():
    # Paths based on your folder structure
    image_path = os.path.join('images', 'sample.jpg')
    encoded_image_path = os.path.join('images', 'encoded_image.png')
    saliency_map_path = os.path.join('images', 'saliency_map.png')

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    print(f"Loading image from path: {image_path}")
    print("Image loaded successfully!")

    print("Computing saliency map...")
    saliency_map = compute_saliency_map(image)
    print(f"Saliency map has {np.sum(saliency_map > 0)} non-zero pixels.")

    # Save and display the saliency map for verification
    cv2.imwrite(saliency_map_path, saliency_map)
    print(f"Saliency map saved at '{saliency_map_path}'")
    display_saliency_map(saliency_map)

    print("Saliency map computed, proceeding with encoding.")
    message = input("Enter the secret message to encode: ")
    encoded_image = encode_message(image, message, saliency_map)

    cv2.imwrite(encoded_image_path, encoded_image)
    print(f"Message encoded and saved as '{encoded_image_path}'")
    
     # Calculate MSE and PSNR
    print("Calculating MSE and PSNR...")
    mse, psnr = calculate_metrics(image, encoded_image)
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")

    print("Decoding message...")
    decoded_message = decode_message(encoded_image, saliency_map, len(message) * 8)
    print(f"Decoded message: {decoded_message}")

if __name__ == '__main__':
    main()
