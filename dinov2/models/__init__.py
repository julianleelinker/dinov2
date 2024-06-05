# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_distill(cfg_student, cfg_teacher, only_teacher=False, img_size=224):
    cfg_student.arch = cfg_student.arch.removesuffix("_memeff")
    assert "vit" in cfg_teacher.arch, '"vit" not in cfg_teacher.arch'
    assert "vit" in cfg_student.arch, '"vit" not in cfg_student.arch'
    teacher_kwargs = dict(
        img_size=img_size,
        patch_size=cfg_teacher.patch_size,
        init_values=cfg_teacher.layerscale,
        ffn_layer=cfg_teacher.ffn_layer,
        block_chunks=cfg_teacher.block_chunks,
        qkv_bias=cfg_teacher.qkv_bias,
        proj_bias=cfg_teacher.proj_bias,
        ffn_bias=cfg_teacher.ffn_bias,
        num_register_tokens=cfg_teacher.num_register_tokens,
        interpolate_offset=cfg_teacher.interpolate_offset,
        interpolate_antialias=cfg_teacher.interpolate_antialias,
    )
    teacher = vits.__dict__[cfg_teacher.arch](**teacher_kwargs)
    if only_teacher:
        return teacher, teacher.embed_dim
    student_kwargs = dict(
        img_size=img_size,
        patch_size=cfg_student.patch_size,
        init_values=cfg_student.layerscale,
        ffn_layer=cfg_student.ffn_layer,
        block_chunks=cfg_student.block_chunks,
        qkv_bias=cfg_student.qkv_bias,
        proj_bias=cfg_student.proj_bias,
        ffn_bias=cfg_student.ffn_bias,
        num_register_tokens=cfg_student.num_register_tokens,
        interpolate_offset=cfg_student.interpolate_offset,
        interpolate_antialias=cfg_student.interpolate_antialias,
    )
    student = vits.__dict__[cfg_student.arch](
        **student_kwargs,
        drop_path_rate=cfg_student.drop_path_rate,
        drop_path_uniform=cfg_student.drop_path_uniform,
    )
    embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    if cfg.distill:
        return build_model_distill(cfg.student, cfg.teacher, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
