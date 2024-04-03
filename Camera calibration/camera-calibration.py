import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.linalg import rq, inv

image_points = [
        [640, 763], [727, 812], [560, 825],
        [1232, 729], [1194, 645], [1108, 754]
    ]

calibration_image_points = [
        [572, 794], [702, 608], [491, 459],
        [1225, 846], [1377, 441], [1098, 667]
    ]

print(calibration_image_points)

point_count = len(calibration_image_points)
calibration_image_points = np.array(calibration_image_points)
fig, ax = plt.subplots(figsize=(10, 15))

calibration_img = mpimg.imread('calibration-rig.jpg')
ax.imshow(calibration_img)
markersize = np.square(15 * np.ones(point_count))
ax.scatter(calibration_image_points[:,0], calibration_image_points[:,1], s=markersize, color='red')
ax.set_title('Calibration Image Points')
plt.show()

world_coordinates = [(10,0,2), (4,0,2), (6,0,0), (4,0,0), (0,4,2), (0,4,0)]
world_coordinates = np.array(world_coordinates)
camera_points = np.array(calibration_image_points)

def calibrate_camera(camera_pts, world_pts):
  assert camera_pts.shape == world_pts.shape[:-1] + (2,)
  points_total = camera_pts.shape[0]
  world_homogeneous = np.hstack((world_pts, np.ones((points_total, 1))))
  camera_homogeneous = np.hstack((camera_pts, np.ones((points_total, 1))))
  matrix_A = np.zeros((2 * points_total, 12))
  for i in range(points_total):
      matrix_A[2 * i] = np.concatenate((np.zeros(4), -world_homogeneous[i], camera_homogeneous[i, 1] * world_homogeneous[i]))
      matrix_A[2 * i + 1] = np.concatenate((world_homogeneous[i], np.zeros(4), -camera_homogeneous[i, 0] * world_homogeneous[i]))
  _, _, V_transpose = np.linalg.svd(matrix_A)
  projection_matrix = V_transpose[-1].reshape((3, 4))
  return projection_matrix

projection_matrix = calibrate_camera(camera_points, world_coordinates)
print(projection_matrix)

def project_world_to_image(projection_mat, world_pts):
  world_ext = np.hstack((world_pts, np.ones((world_pts.shape[0], 1))))
  projected = projection_mat @ world_ext.T
  x_coords = (projected[0, :] / projected[2, :]).reshape((-1, 1))
  y_coords = (projected[1, :] / projected[2, :]).reshape((-1, 1))
  return np.hstack((x_coords, y_coords))

reprojected_points = project_world_to_image(projection_matrix, world_coordinates)
print("Reprojected Image Coordinates:\n\n", reprojected_points)

def compute_reprojection_error(proj_matrix, world_pts, cam_pts):
  reprojected_pts = project_world_to_image(proj_matrix, world_pts)
  error = np.sum((reprojected_pts - cam_pts) ** 2) / reprojected_pts.shape[0]
  return error

reprojection_error = compute_reprojection_error(projection_matrix, world_coordinates, camera_points)
print("Reprojection Error:", reprojection_error)


def extract_parameters(proj_matrix):
  matrix_3x3 = proj_matrix[:, :3]
  intrinsic_matrix, rotation_matrix = rq(matrix_3x3)
  normalization_factor = np.diag(np.sign(np.diag(intrinsic_matrix)))
  if np.linalg.det(normalization_factor) < 0:
    normalization_factor[1, 1] = -1
  intrinsic_matrix_corrected = intrinsic_matrix @ normalization_factor
  rotation_corrected = normalization_factor @ rotation_matrix
  camera_center = inv(-matrix_3x3) @ proj_matrix[:, 3]
  return intrinsic_matrix_corrected, rotation_corrected, camera_center

intrinsic_matrix, rotation_matrix, camera_center = extract_parameters(projection_matrix)
print("Intrinsic Matrix:\n", intrinsic_matrix)
print("\nRotation Matrix:\n", rotation_matrix)
print("\nCamera Center:\n", camera_center)
