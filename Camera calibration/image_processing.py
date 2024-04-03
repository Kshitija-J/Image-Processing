import matplotlib.pyplot as plt
import cv2

def select_points(image_path):
 
  img = cv2.imread(image_path)

  # Convert to RGB for matplotlib display
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Plot the image
  fig, ax = plt.subplots()
  ax.imshow(img)

  # Initialize empty list for selected points
  selected_points = []

  def onclick(event):
    if event.inaxes:
      x, y = event.xdata, event.ydata
      print(f"Click detected at: ({x:.2f}, {y:.2f})")
      selected_points.append((int(x), int(y)))
    
  fig.canvas.mpl_connect('button_press_event', onclick)

  num_points = int(input("Enter the number of points to select: "))

 
  while len(selected_points) < num_points:
    plt.show(block=True)

 
  plt.close()

  return selected_points
