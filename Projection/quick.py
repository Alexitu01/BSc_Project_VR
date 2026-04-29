from PIL import Image
img = Image.open(".\2026-03-19 16_23_58.289016.png")
img.save("panorama_check.jpg")
print(f"Size: {img.size}")