use std::{fs::File, io::Read, path::PathBuf};

use ahash::HashMap;
use glam::{IVec3, UVec3};
use image::DynamicImage;
use ndshape::{RuntimeShape, Shape};
use ron::{
    de::from_reader,
    ser::{to_string_pretty, PrettyConfig},
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use texture_packer::{
    exporter::ImageExporter, texture::Texture, Rect, TexturePacker, TexturePackerConfig,
};
use vinox_voxel::prelude::{
    AssetRegistry, BlockData, BlockRegistry, ChunkData, ChunkPos, GeometryRegistry, RawChunk,
    UVRect, VoxRegistry, Voxel, VoxelPos, CHUNK_SIZE,
};

use crate::raycast::raycast_world;

fn linearize(level_size: impl Into<mint::Vector3<u32>>, pos: glam::UVec3) -> usize {
    let level_size = level_size.into();
    let shape = RuntimeShape::<u32, 3>::new([level_size.x, level_size.y, level_size.z]);
    shape.linearize([pos.x, pos.y, pos.z]) as usize
}

fn to_relative(pos: glam::UVec3) -> glam::UVec3 {
    glam::UVec3::new(
        pos.x.rem_euclid(CHUNK_SIZE as u32),
        pos.y.rem_euclid(CHUNK_SIZE as u32),
        pos.z.rem_euclid(CHUNK_SIZE as u32),
    )
}

fn to_chunk(pos: glam::UVec3) -> glam::UVec3 {
    glam::UVec3::new(
        (pos.x as f32 / (CHUNK_SIZE as f32)).floor() as u32,
        (pos.y as f32 / (CHUNK_SIZE as f32)).floor() as u32,
        (pos.z as f32 / (CHUNK_SIZE as f32)).floor() as u32,
    )
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "V: Serialize + DeserializeOwned, R: Serialize + DeserializeOwned")]
pub struct VoxelLevel<
    V: Voxel<R> + DeserializeOwned = BlockData,
    R: VoxRegistry<V> + DeserializeOwned = BlockRegistry,
> {
    /// Level size in chunks
    pub level_size: mint::Vector3<u32>,
    pub stored_chunks: Option<Vec<RawChunk<V, R>>>,
    pub asset_registry: AssetRegistry,
    pub texture_atlas: Option<Vec<u8>>,
    pub geometry_registry: GeometryRegistry,
    pub block_registry: R,

    #[serde(skip)]
    pub loaded_chunks: Option<Vec<ChunkData<V, R>>>,
}

impl<
        V: Voxel<R> + Clone + Serialize + DeserializeOwned + Eq + Default,
        R: VoxRegistry<V> + Clone + Default + Serialize + DeserializeOwned,
    > VoxelLevel<V, R>
{
    pub fn delinearize(&self, idx: usize) -> glam::UVec3 {
        let shape =
            RuntimeShape::<u32, 3>::new([self.level_size.x, self.level_size.y, self.level_size.z]);
        shape.delinearize(idx as u32).into()
    }

    pub fn linearize(&self, pos: impl Into<mint::Vector3<u32>>) -> u32 {
        let pos: mint::Vector3<u32> = pos.into();
        let shape =
            RuntimeShape::<u32, 3>::new([self.level_size.x, self.level_size.y, self.level_size.z]);
        shape.linearize([pos.x, pos.y, pos.z])
    }

    pub fn new(level_size: impl Into<mint::Vector3<u32>>) -> Self {
        let level_size = level_size.into();
        let mut loaded_chunks = vec![
            ChunkData::<V, R>::default();
            (level_size.x * level_size.y * level_size.z) as usize
        ];
        for x in 0..level_size.x {
            for y in 0..level_size.y {
                for z in 0..level_size.z {
                    let pos = glam::UVec3::new(x, y, z);
                    let idx = linearize(level_size, pos);
                    loaded_chunks.insert(idx, ChunkData::<V, R>::default());
                }
            }
        }
        Self {
            level_size,
            loaded_chunks: Some(loaded_chunks),
            stored_chunks: None,
            asset_registry: AssetRegistry {
                texture_uvs: HashMap::default(),
                texture_size: glam::Vec2::splat(0.0).into(),
            },
            texture_atlas: None,
            geometry_registry: GeometryRegistry::default(),
            block_registry: R::default(),
        }
    }

    pub fn set_voxel(&mut self, voxel_pos: impl Into<mint::Vector3<u32>>, voxel: V) {
        let voxel_pos = voxel_pos.into();
        let chunk_pos = to_chunk(voxel_pos.into());
        let relative_pos: mint::Vector3<u32> = to_relative(voxel_pos.into()).into();

        if let Some(chunk) = self
            .loaded_chunks
            .as_mut()
            .and_then(|x| x.get_mut(linearize(self.level_size, chunk_pos)))
        {
            chunk.set(relative_pos.into(), voxel);
        }
    }

    pub fn get_voxel(&self, voxel_pos: impl Into<mint::Vector3<u32>>) -> Option<V> {
        let voxel_pos = voxel_pos.into();
        let chunk_pos = to_chunk(voxel_pos.into());
        let relative_pos: mint::Vector3<u32> = to_relative(voxel_pos.into()).into();

        self.loaded_chunks
            .as_ref()
            .and_then(|x| x.get(linearize(self.level_size, chunk_pos)))
            .map(|chunk| chunk.get(relative_pos.into()))
    }

    pub fn build_atlas(&mut self, atlas: DynamicImage) {
        self.texture_atlas = Some(atlas.as_bytes().to_vec());
    }

    pub fn load_textures<F>(&mut self, path: PathBuf, closure: F)
    where
        F: Fn(PathBuf) -> Vec<(String, DynamicImage)>,
    {
        let images = closure(path);
        let mut tb = TextureAtlasBuilder::default();
        for (name, image) in images {
            tb.add_texture(name, image);
        }
        let texture_atlas = tb.build().unwrap();
        for (name, uv) in texture_atlas.textures.iter() {
            let rect = rect_to_uv_rect(*uv);
            let identifier = name.split('_').last().unwrap_or_default();
            let final_name = {
                let name = name.clone();
                let pos = name.rfind('_');
                if let Some(pos) = pos {
                    name.get(0..pos).unwrap_or(&name).to_string()
                } else {
                    name
                }
            };
            let mut rects = if let Some(rects) = self
                .asset_registry
                .texture_uvs
                .get_mut(&final_name)
                .copied()
            {
                rects
            } else {
                [rect, rect, rect, rect, rect, rect]
            };

            // If the user named the block_name with anything at the end like _side _front _bottom etc we handle it
            match identifier {
                "side" => {
                    rects[0] = rect;
                    rects[1] = rect;
                    rects[4] = rect;
                    rects[5] = rect;
                }
                "west" => {
                    rects[0] = rect;
                }
                "east" => {
                    rects[1] = rect;
                }
                "south" => {
                    rects[4] = rect;
                }
                "north" => {
                    rects[5] = rect;
                }
                "up" => {
                    rects[3] = rect;
                }
                "down" => {
                    rects[2] = rect;
                }
                &_ => {}
            }
            if let Some(textures) = self.asset_registry.texture_uvs.get_mut(&final_name) {
                *textures = rects;
            } else {
                self.asset_registry.texture_uvs.insert(final_name, rects);
            }
        }

        self.asset_registry.texture_size =
            glam::Vec2::new(texture_atlas.size.x as f32, texture_atlas.size.y as f32).into();
        self.texture_atlas = Some(texture_atlas.image.as_bytes().to_vec());
    }

    pub fn save(&mut self, path: PathBuf, name: String) {
        if let Some(loaded_chunks) = self.loaded_chunks.clone() {
            self.stored_chunks = Some(loaded_chunks.iter().map(|x| x.to_raw()).collect());
        }
        let pretty = PrettyConfig::new()
            .depth_limit(2)
            .separate_tuple_members(true)
            .enumerate_arrays(true);
        let s = to_string_pretty(&self, pretty).expect("Serialization failed");
        let final_path = path.join(name);
        std::fs::write(final_path, s).expect("Couldn't write file");
    }

    pub fn load(path: PathBuf, name: String) -> Self {
        let file = File::open(path.join(name)).expect("Couldn't open file");
        match from_reader::<File, Self>(file) {
            Ok(mut x) => {
                x.loaded_chunks = Some(
                    x.stored_chunks
                        .clone()
                        .unwrap_or_default()
                        .iter()
                        .map(|x| ChunkData::from_raw(x.clone()))
                        .collect(),
                );
                x
            }
            Err(e) => {
                println!("Failed to load level: {}", e);

                std::process::exit(1);
            }
        }
    }

    pub fn load_buf_reader(&self, reader: impl Read) {
        match from_reader(reader) {
            Ok(x) => x,
            Err(e) => {
                println!("Failed to load level: {}", e);

                std::process::exit(1);
            }
        }
    }

    pub fn raycast(
        &self,
        origin: impl Into<mint::Vector3<f32>>,
        direction: impl Into<mint::Vector3<f32>>,
        radius: f32,
    ) -> Option<(VoxelPos, mint::Vector3<f32>, f32)> {
        raycast_world(origin, direction, radius, |vox_pos| {
            let vox_pos: UVec3 = IVec3::from(mint::Vector3::<i32>::from(vox_pos)).as_uvec3();
            if let Some(voxel) = self.get_voxel(vox_pos) {
                !voxel.is_empty(Some(&self.block_registry))
            } else {
                false
            }
        })
    }

    pub fn get_chunk(&self, chunk_pos: impl Into<mint::Vector3<u32>>) -> Option<&ChunkData<V, R>> {
        let chunk_pos: mint::Vector3<u32> = chunk_pos.into();
        let chunk_pos = UVec3::from(chunk_pos);
        self.loaded_chunks
            .as_ref()
            .and_then(|x| x.get(linearize(self.level_size, chunk_pos)))
    }

    pub fn get_chunk_neighbors_pos(
        &self,
        chunk_pos: impl Into<mint::Vector3<u32>>,
    ) -> Option<[Option<UVec3>; 26]> {
        let chunk_pos: mint::Vector3<u32> = chunk_pos.into();
        let chunk_pos = UVec3::from(chunk_pos);
        let vox_neighbors =
            ChunkPos::new(chunk_pos.x as i32, chunk_pos.y as i32, chunk_pos.z as i32).neighbors();

        self.loaded_chunks.as_ref().and_then(|x| {
            let mut neighbors = Vec::default();
            for neighbor in vox_neighbors {
                if neighbor.x >= 0 && neighbor.y >= 0 && neighbor.z >= 0
                // && neighbor.lt(&UVec3::from(self.level_size).as_ivec3().into())
                {
                    if x.get(linearize(
                        self.level_size,
                        UVec3::new(neighbor.x as u32, neighbor.y as u32, neighbor.z as u32),
                    ))
                    .is_some()
                    {
                        neighbors.push(Some(UVec3::new(
                            neighbor.x as u32,
                            neighbor.y as u32,
                            neighbor.z as u32,
                        )));
                    } else {
                        neighbors.push(None)
                    }
                } else {
                    neighbors.push(None)
                }
            }
            neighbors.try_into().ok()
        })
    }

    pub fn get_chunk_neighbors_cloned(
        &self,
        chunk_pos: impl Into<mint::Vector3<u32>>,
    ) -> Option<[ChunkData<V, R>; 26]> {
        let chunk_pos: mint::Vector3<u32> = chunk_pos.into();
        let chunk_pos = UVec3::from(chunk_pos);
        let vox_neighbors =
            ChunkPos::new(chunk_pos.x as i32, chunk_pos.y as i32, chunk_pos.z as i32).neighbors();
        self.loaded_chunks.as_ref().and_then(|x| {
            let mut neighbors = Vec::default();
            for neighbor in vox_neighbors {
                if neighbor.x >= 0 && neighbor.y >= 0 && neighbor.z >= 0
                // && neighbor.lt(&UVec3::from(self.level_size).as_ivec3().into())
                {
                    if let Some(chunk) = x.get(linearize(
                        self.level_size,
                        UVec3::new(neighbor.x as u32, neighbor.y as u32, neighbor.z as u32),
                    )) {
                        neighbors.push(chunk.clone());
                    } else {
                        neighbors.push(ChunkData::<V, R>::default())
                    }
                } else {
                    neighbors.push(ChunkData::<V, R>::default())
                }
            }
            neighbors.try_into().ok()
        })
    }
}

struct TextureAtlasBuilder<H: std::hash::Hash> {
    images: HashMap<H, DynamicImage>,
    packer_conf: TexturePackerConfig,
}

impl Default for TextureAtlasBuilder<String> {
    fn default() -> Self {
        let config = TexturePackerConfig {
            max_width: 1024,
            max_height: 1024,
            allow_rotation: false,
            texture_outlines: false,
            border_padding: 2,
            texture_padding: 2,
            ..Default::default()
        };
        Self {
            packer_conf: config,
            images: HashMap::default(),
        }
    }
}

struct TextureAtlas<H: std::hash::Hash> {
    image: DynamicImage,
    size: mint::Point2<u32>,
    textures: HashMap<H, Rect>,
}

impl<H: std::hash::Hash + std::cmp::Eq + std::clone::Clone> TextureAtlasBuilder<H> {
    fn add_texture(&mut self, hash: H, image: DynamicImage) {
        self.images.insert(hash, image);
    }

    fn build(&mut self) -> Result<TextureAtlas<H>, ()> {
        let mut packer = TexturePacker::new_skyline(self.packer_conf);
        for (hash, image) in self.images.iter() {
            let pixels = image.as_bytes().to_vec();
            let img = image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(image.width(), image.height(), pixels).unwrap(),
            );
            packer.pack_own(hash.clone(), img).unwrap();
        }
        let mut textures = HashMap::default();
        for (hash, frame) in packer.get_frames() {
            let rect = Rect {
                x: frame.frame.x,
                y: frame.frame.y,
                w: frame.frame.w,
                h: frame.frame.h,
            };
            textures.insert(hash.clone(), rect);
        }

        let exporter = ImageExporter::export(&packer).unwrap();
        let final_img = exporter.into_rgba8();

        Ok(TextureAtlas {
            image: DynamicImage::from(final_img),
            size: mint::Point2 {
                x: packer.width(),
                y: packer.height(),
            },
            textures,
        })
    }
}

fn rect_to_uv_rect(rect: Rect) -> UVRect {
    UVRect {
        x: rect.x as f32,
        y: rect.y as f32,
        w: rect.w as f32,
        h: rect.h as f32,
    }
}
