import matplotlib.pyplot as plt
import image_processing  

def main():
  image_path = "calibration-rig.jpg" 
  selected_points = image_processing.select_points(image_path)

  for point in selected_points:
    print(f"Selected point: ({point[0]}, {point[1]})")
 
if __name__ == "__main__":
  main()
