import pinocchio as pin
import numpy as np

URDF_PATH = "assets/franka_panda/robots/panda_arm.urdf"

model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()

print("model loaded ok")
print("nq =", model.nq)
print("nv =", model.nv)

q = pin.neutral(model)

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

print("\nframes:")
for i, f in enumerate(model.frames):
    print(i, f.name)

frame_name = "panda_link8"
frame_id = model.getFrameId(frame_name)

J = pin.computeFrameJacobian(
    model,
    data,
    q,
    frame_id,
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)

print("\nend-effector frame:", frame_name)
print("frame id:", frame_id)
print("jacobian shape:", J.shape)
print("ee translation:", data.oMf[frame_id].translation.T)
print("jacobian:\n", J)