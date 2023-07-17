use glam::Vec3;
use mint::Vector3;
use vinox_voxel::prelude::{ChunkPos, RelativeVoxelPos, VoxelPos};

// Takes in absolute world positions returns a chunk pos and a voxel pos for whatever face it hits and a normal
pub fn raycast_world(
    origin: impl Into<mint::Vector3<f32>>,
    direction: impl Into<mint::Vector3<f32>>,
    radius: f32,
    block_callback: impl Fn(VoxelPos) -> bool,
) -> Option<(ChunkPos, RelativeVoxelPos, Vec3, f32)> {
    let direction: Vector3<f32> = direction.into();
    let origin: Vector3<f32> = origin.into();
    let direction: Vec3 = direction.into();
    let origin: Vec3 = origin.into();

    // TMax needs the fractional part of origin to work.
    let mut tmax = Vec3::new(
        intbound(origin.x, direction.x),
        intbound(origin.y, direction.y),
        intbound(origin.z, direction.z),
    );

    let mut current_block: Vec3 = Vector3::from(VoxelPos::from(Vector3::from(origin))).into();
    let step = direction.signum();

    let tdelta = step / direction;

    let mut face = Vec3::ZERO;

    if direction == Vec3::ZERO {
        return None;
    }

    let radius = radius
        / (direction.x * direction.x + direction.y * direction.y + direction.z * direction.z)
            .sqrt();
    let mut lastmax = 0.0;
    let mut counter = 0;
    loop {
        // Infinite loop shouldve been prevented by tmax but it isn't for some reason all the time. This just breaks the loop after 4 x the radius which should be plenty of time to find the voxel
        // It could be due to chunks or neighbors not existing?
        if counter > (radius * 4.0) as u32 {
            break;
        }
        let voxel_pos = VoxelPos::from(Vector3::from(current_block));
        if block_callback(voxel_pos) {
            let (relative_pos, chunk_pos) = voxel_pos.to_offsets();
            let toi = lastmax * direction.length();
            return Some((chunk_pos, relative_pos, face, toi));
        }

        if tmax.x < tmax.y {
            if tmax.x < tmax.z {
                if tmax.x > radius {
                    break;
                }
                lastmax = tmax.x;
                current_block.x += step.x;
                tmax.x += tdelta.x;
                face.x = -step.x;
                face.y = 0.0;
                face.z = 0.0;
            } else {
                if tmax.z > radius {
                    break;
                }
                lastmax = tmax.z;
                current_block.z += step.z;
                tmax.z += tdelta.z;
                face.x = 0.0;
                face.y = 0.0;
                face.z = -step.z;
            }
        } else if tmax.y < tmax.z {
            if tmax.y > radius {
                break;
            }
            lastmax = tmax.y;
            current_block.y += step.y;
            tmax.y += tdelta.y;
            face.x = 0.0;
            face.y = -step.y;
            face.z = 0.0;
        } else {
            if tmax.z > radius {
                break;
            }
            lastmax = tmax.z;
            current_block.z += step.z;
            tmax.z += tdelta.z;
            face.x = 0.0;
            face.y = 0.0;
            face.z = -step.z;
        }
        counter += 1;
    }
    None
}

fn intbound(s: f32, ds: f32) -> f32 {
    if ds < 0.0 {
        intbound(-s, -ds)
    } else {
        let s = s.rem_euclid(1.0);
        (1.0 - s) / ds
    }
}
