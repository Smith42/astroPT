def create_dataloaders(
    config: TrainingConfig, 
    ddp: bool
) -> Tuple[DataLoader, DataLoader, ModalityRegistry]:
    """
    Initializes Datasets and DataLoaders for the training pipeline.

    This function acts as a bridge, translating the flat 'TrainingConfig'
    hyperparameters into the structured 'ModalityRegistry' objects required by 
    both the Model and the Dataset.

    Args:
        config (TrainingConfig): The global configuration object containing paths and params.
        ddp (bool): Flag indicating if Distributed Data Parallelism is active.

    Returns:
        Tuple[DataLoader, DataLoader, ModalityRegistry]: A tuple containing:
            - train_loader: The DataLoader for the training set.
            - val_loader: The DataLoader for the validation/test set.
            - registry: The configured ModalityRegistry (needed to initialize the GPT model).
    """
    
    # Configure ModalityRegistry
    modalities = []
    tf_kwargs = {}
    
    # Define configuration for each modality
    if config.images_train:
        if config.images_tokenizer_method == "discrete":
            # Discrete (AION) tokenization: tokens are integer IDs
            image_modality_config = ModalityConfig(
                name="aion_images",
                input_size=1,                           # Not used (Embedder uses vocab_size)
                patch_size=1,                           # Each token is a single ID
                pos_input_size=config.images_pos_input_size,
                loss_weight=config.images_loss_weight,
                embed_pos=True,                         # Learnt position embeddings
                vocab_size=config.images_tokeniser_discrete_vocab_size, # FSQ levels [8,8,8,5,5,5]
                encoder_type="discrete",
            )
        else:
            # Continuous regression: patches of pixel values
            img_input_batch_size = config.images_patch_size * config.images_patch_size * config.images_channels
            image_modality_config = ModalityConfig(
                name="images",
                input_size=img_input_batch_size,
                patch_size=config.images_patch_size,
                pos_input_size=config.images_pos_input_size,  
                loss_weight=config.images_loss_weight,
                embed_pos=config.images_embed_pos,
                encoder_type=config.images_tokenizer_method, # "aim" or "affine"
            )
        modalities.append(image_modality_config)
        
        # Transforms (always needed as AION expects normalized inputs)
        tf_kwargs.update({
            'norm_type_img': config.images_norm_type,
            'norm_scaler_img': config.images_norm_scaler,
            'norm_const_img': config.images_norm_const,
        })
        
    if config.spectra_train:
        if config.spectra_tokenizer_method == "discrete":
            # Discrete (AION) tokenization: tokens are integer IDs
            spectra_modality_config = ModalityConfig(
                name="aion_spectra",
                input_size=1,                           # Not used (Embedder uses vocab_size)
                patch_size=1,                           # Each token is a single ID
                pos_input_size=config.spectra_pos_input_size,
                loss_weight=config.spectra_loss_weight,
                embed_pos=True,                         # Learnt position embeddings
                vocab_size=config.spectra_tokeniser_discrete_vocab_size, # LFQ codebook_size
                encoder_type="discrete",
            )
        else:
            # Continuous regression: patches of spectral flux
            spectra_modality_config = ModalityConfig(
                name="spectra",
                input_size=config.spectra_patch_size,
                patch_size=config.spectra_patch_size,
                pos_input_size=config.spectra_pos_input_size, 
                loss_weight=config.spectra_loss_weight,
                embed_pos=config.spectra_embed_pos,
                encoder_type=config.spectra_tokenizer_method, # "aim" or "affine"
            )
        modalities.append(spectra_modality_config)
        
        # Transforms (always needed as AION expects normalized inputs)
        tf_kwargs.update({
            'norm_type_spec': config.spectra_norm_type,
            'norm_scaler_spec': config.spectra_norm_scaler,
            'norm_const_spec': config.spectra_norm_const,
        })
    
    # Instantiate the Registry
    registry = ModalityRegistry(modalities)
    
    # Use data augmentation for training
    train_stage = 'train' if config.use_aug else 'val'
        
    # 4. Instantiate transforms dynamically unpacking the dictionary
    train_tf = EuclidDESIDatasetArrow.data_transforms(
        stage=train_stage, 
        **tf_kwargs
    )
    
    val_tf = EuclidDESIDatasetArrow.data_transforms(
        stage='val', 
        **tf_kwargs
    )
    
    # Activating the logger object
    logger = logging.getLogger("AstroPT")
    
    # Informational log (Only printed by the Master Process to avoid spam)
    if not ddp or (ddp and int(os.environ.get("RANK", 0)) == 0):
        logger.info(f"Loading data from: {config.data_dir}")

    # Instantiate Train Dataset 
    train_dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="train",
        modality_registry=registry, 
        spiral=config.spiral,
        stochastic=True,
        transform=train_tf,
        spectra_inverse=config.spectra_inverse,
        spectra_mask=config.spectra_mask,
        spectra_mask_prob=config.spectra_mask_prob,
        images_mask=config.images_mask,
        images_mask_prob=config.images_mask_prob,
        unet_weights_path=config.images_unet_weights_path,
        aion_image_size=config.images_aion_image_size,
        aion_image_transform=config.images_aion_image_transform,
        use_pretokenized=config.use_pretokenized,
    )
    
    # Instantiate Validation/Test Dataset 
    val_dataset = EuclidDESIDatasetArrow(
        arrow_folder_root=config.data_dir,
        split="test", 
        modality_registry=registry,
        spiral=config.spiral,
        stochastic=False,
        transform=val_tf,
        spectra_inverse=config.spectra_inverse,
        spectra_mask=config.spectra_mask,
        spectra_mask_prob=config.spectra_mask_prob,
        images_mask=config.images_mask,
        images_mask_prob=config.images_mask_prob,
        unet_weights_path=config.images_unet_weights_path,
        aion_image_size=config.images_aion_image_size,
        aion_image_transform=config.images_aion_image_transform,
        use_pretokenized=config.use_pretokenized,
    )

    # Configure DDP Samplers
    if ddp:
        # In DDP the sampler splits the data among GPUs
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create Final DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )

    return train_loader, val_loader, registry
