{
    "ema": {
      "desc": null,
      "value": {
        "power": 0.75,
        "_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel",
        "inv_gamma": 1,
        "max_value": 0.9999,
        "min_value": 0,
        "update_after_step": 0
      }
    },
    "name": {
      "desc": null,
      "value": "train_diffusion_unet_image"
    },
    "task": {
      "desc": null,
      "value": {
        "name": "real_image",
        "dataset": {
          "seed": 42,
          "horizon": 16,
          "_target_": "diffusion_policy.dataset.real_pusht_image_dataset.RealPushTImageDataset",
          "pad_after": 7,
          "use_cache": true,
          "val_ratio": 0,
          "pad_before": 1,
          "shape_meta": {
            "obs": {
              "camera_0": {
                "type": "rgb",
                "shape": [
                  3,
                  240,
                  320
                ]
              },
              "camera_1": {
                "type": "rgb",
                "shape": [
                  3,
                  240,
                  320
                ]
              },
              "camera_2": {
                "type": "rgb",
                "shape": [
                  3,
                  240,
                  320
                ]
              },
              "robot_eef_pose": {
                "type": "low_dim",
                "shape": [
                  6
                ]
              }
            },
            "action": {
              "shape": [
                6
              ]
            }
          },
          "n_obs_steps": 2,
          "dataset_path": "data/demo_pusht_real",
          "delta_action": false,
          "n_latency_steps": 0,
          "max_train_episodes": null
        },
        "env_runner": {
          "_target_": "diffusion_policy.env_runner.real_pusht_image_runner.RealPushTImageRunner"
        },
        "shape_meta": {
          "obs": {
            "camera_0": {
              "type": "rgb",
              "shape": [
                3,
                240,
                320
              ]
            },
            "camera_1": {
              "type": "rgb",
              "shape": [
                3,
                240,
                320
              ]
            },
            "camera_2": {
              "type": "rgb",
              "shape": [
                3,
                240,
                320
              ]
            },
            "robot_eef_pose": {
              "type": "low_dim",
              "shape": [
                6
              ]
            }
          },
          "action": {
            "shape": [
              6
            ]
          }
        },
        "image_shape": [
          3,
          240,
          320
        ],
        "dataset_path": "data/demo_pusht_real"
      }
    },
    "_wandb": {
      "desc": null,
      "value": {
        "t": {
          "1": [
            1,
            41,
            49,
            50,
            55,
            71
          ],
          "2": [
            1,
            41,
            49,
            50,
            55,
            71
          ],
          "3": [
            13,
            15,
            16,
            23
          ],
          "4": "3.9.15",
          "5": "0.13.3",
          "8": [
            5
          ]
        },
        "framework": "torch",
        "start_time": 1699039560.490738,
        "cli_version": "0.13.3",
        "is_jupyter_run": false,
        "python_version": "3.9.15",
        "is_kaggle_kernel": false
      }
    },
    "policy": {
      "desc": null,
      "value": {
        "horizon": 16,
        "_target_": "diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy",
        "n_groups": 8,
        "down_dims": [
          512,
          1024,
          2048
        ],
        "shape_meta": {
          "obs": {
            "camera_0": {
              "type": "rgb",
              "shape": [
                3,
                240,
                320
              ]
            },
            "camera_1": {
              "type": "rgb",
              "shape": [
                3,
                240,
                320
              ]
            },
            "camera_2": {
              "type": "rgb",
              "shape": [
                3,
                240,
                320
              ]
            },
            "robot_eef_pose": {
              "type": "low_dim",
              "shape": [
                6
              ]
            }
          },
          "action": {
            "shape": [
              6
            ]
          }
        },
        "kernel_size": 5,
        "n_obs_steps": 2,
        "obs_encoder": {
          "_target_": "diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder",
          "rgb_model": {
            "name": "resnet18",
            "weights": null,
            "_target_": "diffusion_policy.model.vision.model_getter.get_resnet"
          },
          "crop_shape": [
            216,
            288
          ],
          "shape_meta": {
            "obs": {
              "camera_0": {
                "type": "rgb",
                "shape": [
                  3,
                  240,
                  320
                ]
              },
              "camera_1": {
                "type": "rgb",
                "shape": [
                  3,
                  240,
                  320
                ]
              },
              "camera_2": {
                "type": "rgb",
                "shape": [
                  3,
                  240,
                  320
                ]
              },
              "robot_eef_pose": {
                "type": "low_dim",
                "shape": [
                  6
                ]
              }
            },
            "action": {
              "shape": [
                6
              ]
            }
          },
          "random_crop": true,
          "resize_shape": [
            240,
            320
          ],
          "imagenet_norm": true,
          "use_group_norm": true,
          "share_rgb_model": false
        },
        "n_action_steps": 8,
        "noise_scheduler": {
          "_target_": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
          "beta_end": 0.02,
          "beta_start": 0.0001,
          "clip_sample": true,
          "steps_offset": 0,
          "beta_schedule": "squaredcos_cap_v2",
          "prediction_type": "epsilon",
          "set_alpha_to_one": true,
          "num_train_timesteps": 100
        },
        "cond_predict_scale": true,
        "obs_as_global_cond": true,
        "num_inference_steps": 100,
        "diffusion_step_embed_dim": 128
      }
    },
    "horizon": {
      "desc": null,
      "value": 16
    },
    "logging": {
      "desc": null,
      "value": {
        "id": null,
        "mode": "online",
        "name": "2023.11.03-14.25.42_train_diffusion_unet_image_real_image",
        "tags": [
          "train_diffusion_unet_image",
          "real_image",
          "default"
        ],
        "group": null,
        "resume": true,
        "project": "diffusion_policy_debug"
      }
    },
    "_target_": {
      "desc": null,
      "value": "diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace"
    },
    "exp_name": {
      "desc": null,
      "value": "default"
    },
    "training": {
      "desc": null,
      "value": {
        "seed": 42,
        "debug": false,
        "device": "cuda:0",
        "resume": true,
        "use_ema": true,
        "val_every": 1,
        "num_epochs": 600,
        "lr_scheduler": "cosine",
        "sample_every": 5,
        "max_val_steps": null,
        "rollout_every": 50,
        "freeze_encoder": false,
        "lr_warmup_steps": 500,
        "max_train_steps": null,
        "checkpoint_every": 50,
        "tqdm_interval_sec": 1,
        "gradient_accumulate_every": 1
      }
    },
    "multi_run": {
      "desc": null,
      "value": {
        "run_dir": "data/outputs/2023.11.03/14.25.42_train_diffusion_unet_image_real_image",
        "wandb_name_base": "2023.11.03-14.25.42_train_diffusion_unet_image_real_image"
      }
    },
    "optimizer": {
      "desc": null,
      "value": {
        "lr": 0.0001,
        "eps": 1e-8,
        "betas": [
          0.95,
          0.999
        ],
        "_target_": "torch.optim.AdamW",
        "weight_decay": 0.000001
      }
    },
    "task_name": {
      "desc": null,
      "value": "real_image"
    },
    "checkpoint": {
      "desc": null,
      "value": {
        "topk": {
          "k": 5,
          "mode": "min",
          "format_str": "epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt",
          "monitor_key": "train_loss"
        },
        "save_last_ckpt": true,
        "save_last_snapshot": false
      }
    },
    "dataloader": {
      "desc": null,
      "value": {
        "shuffle": true,
        "batch_size": 64,
        "pin_memory": true,
        "num_workers": 8,
        "persistent_workers": true
      }
    },
    "output_dir": {
      "desc": null,
      "value": "/home/user/miles/rpm_diffusion_policy/data/outputs/2023.11.03/14.25.42_train_diffusion_unet_image_real_image"
    },
    "shape_meta": {
      "desc": null,
      "value": {
        "obs": {
          "camera_0": {
            "type": "rgb",
            "shape": [
              3,
              240,
              320
            ]
          },
          "camera_1": {
            "type": "rgb",
            "shape": [
              3,
              240,
              320
            ]
          },
          "camera_2": {
            "type": "rgb",
            "shape": [
              3,
              240,
              320
            ]
          },
          "robot_eef_pose": {
            "type": "low_dim",
            "shape": [
              6
            ]
          }
        },
        "action": {
          "shape": [
            6
          ]
        }
      }
    },
    "n_obs_steps": {
      "desc": null,
      "value": 2
    },
    "n_action_steps": {
      "desc": null,
      "value": 8
    },
    "val_dataloader": {
      "desc": null,
      "value": {
        "shuffle": false,
        "batch_size": 64,
        "pin_memory": true,
        "num_workers": 8,
        "persistent_workers": true
      }
    },
    "n_latency_steps": {
      "desc": null,
      "value": 0
    },
    "dataset_obs_steps": {
      "desc": null,
      "value": 2
    },
    "obs_as_global_cond": {
      "desc": null,
      "value": true
    },
    "past_action_visible": {
      "desc": null,
      "value": false
    },
    "keypoint_visible_rate": {
      "desc": null,
      "value": 1
    }
  }