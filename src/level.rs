use std::path::PathBuf;

use ahash::HashMap;
use image::DynamicImage;
use serde::Serialize;
use texture_packer::{
    exporter::ImageExporter, texture::Texture, Rect, TexturePacker, TexturePackerConfig,
};
use vinox_voxel::prelude::{
    AssetRegistry, BlockData, BlockRegistry, ChunkData, RawChunk, UVRect, VoxRegistry, Voxel,
    CHUNK_SIZE,
};

pub struct VoxelLevel<
    V: Voxel<R> + Clone + Serialize + Eq + Default = BlockData,
    R: VoxRegistry<V> + Clone + Default = BlockRegistry,
> {
    /// Level size in chunks
    pub level_size: mint::Vector3<u32>,
    pub loaded_chunks: Option<HashMap<mint::Vector3<u32>, ChunkData<V, R>>>,
    pub stored_chunks: Option<HashMap<mint::Vector3<u32>, RawChunk<V, R>>>,
    pub asset_registry: AssetRegistry,
    pub texture_atlas: Option<Vec<u8>>,
}

impl<V: Voxel<R> + Clone + Serialize + Eq + Default, R: VoxRegistry<V> + Clone + Default>
    VoxelLevel<V, R>
{
    pub fn new(level_size: impl Into<mint::Vector3<u32>>) -> Self {
        let level_size = level_size.into();
        let mut loaded_chunks = HashMap::default();
        for x in 0..level_size.x {
            for y in 0..level_size.y {
                for z in 0..level_size.z {
                    let pos = glam::UVec3::new(x, y, z);
                    loaded_chunks.insert(pos.into(), ChunkData::<V, R>::default());
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
        }
    }

    pub fn set_voxel(&mut self, voxel_pos: impl Into<mint::Vector3<u32>>, voxel: V) {
        let voxel_pos = voxel_pos.into();
        let chunk_pos = glam::UVec3::new(
            (voxel_pos.x as f32 / (CHUNK_SIZE as f32)).floor() as u32,
            (voxel_pos.y as f32 / (CHUNK_SIZE as f32)).floor() as u32,
            (voxel_pos.z as f32 / (CHUNK_SIZE as f32)).floor() as u32,
        )
        .into();

        let relative_pos: mint::Vector3<u32> = glam::UVec3::new(
            voxel_pos.x.rem_euclid(CHUNK_SIZE as u32) as u32,
            voxel_pos.y.rem_euclid(CHUNK_SIZE as u32) as u32,
            voxel_pos.z.rem_euclid(CHUNK_SIZE as u32) as u32,
        )
        .into();

        if let Some(chunk) = self
            .loaded_chunks
            .as_mut()
            .and_then(|x| x.get_mut(&chunk_pos))
        {
            chunk.set(relative_pos.into(), voxel);
        }
    }

    pub fn get_voxel(&mut self, voxel_pos: impl Into<mint::Vector3<u32>>) -> Option<V> {
        let voxel_pos = voxel_pos.into();
        let chunk_pos = glam::UVec3::new(
            (voxel_pos.x as f32 / (CHUNK_SIZE as f32)).floor() as u32,
            (voxel_pos.y as f32 / (CHUNK_SIZE as f32)).floor() as u32,
            (voxel_pos.z as f32 / (CHUNK_SIZE as f32)).floor() as u32,
        )
        .into();

        let relative_pos: mint::Vector3<u32> = glam::UVec3::new(
            voxel_pos.x.rem_euclid(CHUNK_SIZE as u32) as u32,
            voxel_pos.y.rem_euclid(CHUNK_SIZE as u32) as u32,
            voxel_pos.z.rem_euclid(CHUNK_SIZE as u32) as u32,
        )
        .into();

        if let Some(chunk) = self.loaded_chunks.as_ref().and_then(|x| x.get(&chunk_pos)) {
            Some(chunk.get(relative_pos.into()))
        } else {
            None
        }
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
            let mut rects =
                if let Some(rects) = self.asset_registry.texture_uvs.get_mut(&final_name) {
                    rects.clone()
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
                x: frame.frame.x as u32,
                y: frame.frame.y as u32,
                w: frame.frame.w as u32,
                h: frame.frame.h as u32,
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
