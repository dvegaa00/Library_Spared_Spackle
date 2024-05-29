API
===
Import SpaRED::

    import spared



Processing
~~~~~~~~~~~~~~~

.. module:: spared.processing
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    processing.get_slide_from_collection
    processing.get_exp_frac
    processing.get_glob_exp_frac
    processing.filter_dataset
    processing.tpm_normalization
    processing.log1p_transformation
    processing.get_spatial_neighbors
    processing.clean_noise
    processing.combat_transformation
    processing.get_deltas
    processing.compute_moran
    processing.filter_by_moran
    processing.add_noisy_layer
    processing.process_dataset
    processing.compute_patches_embeddings_and_predictions
    processing.get_pretrain_dataloaders
    processing.get_graphs_one_slide
    processing.get_sin_cos_positional_embeddings
    processing.get_graphs
    processing.get_graph_dataloaders


Metrics
~~~~~~~~~~~~~~~
.. module:: spared.metrics
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    metrics.get_metrics

Plotting
~~~~~~~~~~~~~~~
.. module:: spared.visualize
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    visualize.refine_plotting_slides_str
    visualize.get_plotting_slides_adata
    visualize.plot_all_slides
    visualize.get_exp_frac
    visualize.get_glob_exp_frac
    visualize.plot_exp_frac
    visualize.plot_histograms
    visualize.plot_random_patches
    visualize.visualize_moran_filtering
    visualize.visualize_gene_expression
    visualize.compute_dim_red
    visualize.plot_clusters
    visualize.plot_mean_std
    visualize.plot_data_distribution_stats
    visualize.plot_mean_std_partitions
    visualize.plot_tests
