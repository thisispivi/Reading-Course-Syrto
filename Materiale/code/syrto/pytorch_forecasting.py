import torch
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, RecurrentNetwork, LSTM, GRU, DeepAR
from pytorch_forecasting import MAPE, MASE, MAE, MAPE, RMSE, QuantileLoss, MultiLoss, NormalDistributionLoss, SMAPE
from tqdm import tqdm
from pytorch_forecasting.utils import move_to_device, create_mask
from pytorch_forecasting.models.base_model import _torch_cat_na, _concatenate_output


# Adapted from `forward` method of `TemporalFusionTransformer` 
# https://github.com/jdb78/pytorch-forecasting/blob/7e376020871b330ac7c92ca4ff68ad430adee203/pytorch_forecasting/models/temporal_fusion_transformer/__init__.py#L494
def tft_embedding_tensor(tft, x):
    """
    Return the TFT output just before the final linear output layers.
    
    Input dimensions: n_samples x time x variables
    """
    encoder_lengths = x["encoder_lengths"]
    decoder_lengths = x["decoder_lengths"]
    x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
    x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
    timesteps = x_cont.size(1)  # encode + decode length
    max_encoder_length = int(encoder_lengths.max())
    input_vectors = tft.input_embeddings(x_cat)
    input_vectors.update(
        {
            name: x_cont[..., idx].unsqueeze(-1)
            for idx, name in enumerate(tft.hparams.x_reals)
            if name in tft.reals
        }
    )

    # Embedding and variable selection
    if len(tft.static_variables) > 0:
        # static embeddings will be constant over entire batch
        static_embedding = {name: input_vectors[name][:, 0] for name in tft.static_variables}
        static_embedding, static_variable_selection = tft.static_variable_selection(static_embedding)
    else:
        static_embedding = torch.zeros(
            (x_cont.size(0), tft.hparams.hidden_size), dtype=tft.dtype, device=tft.device
        )
        static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=tft.dtype, device=tft.device)

    static_context_variable_selection = tft.expand_static_context(
        tft.static_context_variable_selection(static_embedding), timesteps
    )

    embeddings_varying_encoder = {
        name: input_vectors[name][:, :max_encoder_length] for name in tft.encoder_variables
    }
    embeddings_varying_encoder, encoder_sparse_weights = tft.encoder_variable_selection(
        embeddings_varying_encoder,
        static_context_variable_selection[:, :max_encoder_length],
    )

    embeddings_varying_decoder = {
        name: input_vectors[name][:, max_encoder_length:] for name in tft.decoder_variables  # select decoder
    }
    embeddings_varying_decoder, decoder_sparse_weights = tft.decoder_variable_selection(
        embeddings_varying_decoder,
        static_context_variable_selection[:, max_encoder_length:],
    )

    # LSTM
    # calculate initial state
    input_hidden = tft.static_context_initial_hidden_lstm(static_embedding).expand(
        tft.hparams.lstm_layers, -1, -1
    )
    input_cell = tft.static_context_initial_cell_lstm(static_embedding).expand(tft.hparams.lstm_layers, -1, -1)

    # run local encoder
    encoder_output, (hidden, cell) = tft.lstm_encoder(
        embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
    )

    # run local decoder
    decoder_output, _ = tft.lstm_decoder(
        embeddings_varying_decoder,
        (hidden, cell),
        lengths=decoder_lengths,
        enforce_sorted=False,
    )

    # skip connection over lstm
    lstm_output_encoder = tft.post_lstm_gate_encoder(encoder_output)
    lstm_output_encoder = tft.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

    lstm_output_decoder = tft.post_lstm_gate_decoder(decoder_output)
    lstm_output_decoder = tft.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

    lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

    # static enrichment
    static_context_enrichment = tft.static_context_enrichment(static_embedding)
    attn_input = tft.static_enrichment(
        lstm_output, tft.expand_static_context(static_context_enrichment, timesteps)
    )

    # Attention
    attn_output, attn_output_weights = tft.multihead_attn(
        q=attn_input[:, max_encoder_length:],  # query only for predictions
        k=attn_input,
        v=attn_input,
        mask=tft.get_attention_mask(
            encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length
        ),
    )

    # skip connection over attention
    attn_output = tft.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

    output = tft.pos_wise_ff(attn_output)

    # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
    # a skip from the variable selection network)
    output = tft.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
    
    #####
    return output
    # return tft.to_network_output(
    #     prediction=output,
    #     attention=attn_output_weights,
    #     static_variables=static_variable_selection,
    #     encoder_variables=encoder_sparse_weights,
    #     decoder_variables=decoder_sparse_weights,
    #     decoder_lengths=decoder_lengths,
    #     encoder_lengths=encoder_lengths)
    ######

    # if tft.n_targets > 1:  # if to use multi-target architecture
    #     output = [output_layer(output) for output_layer in tft.output_layer]
    # else:
    #     output = tft.output_layer(output)

    # return tft.to_network_output(
    #     prediction=tft.transform_output(output, target_scale=x["target_scale"]),
    #     attention=attn_output_weights,
    #     static_variables=static_variable_selection,
    #     encoder_variables=encoder_sparse_weights,
    #     decoder_variables=decoder_sparse_weights,
    #     decoder_lengths=decoder_lengths,
    #     encoder_lengths=encoder_lengths,
    # )


def embedding_tensor(model, x):
    if isinstance(model, TemporalFusionTransformer):
        return tft_embedding_tensor(model, x)
    else:
        raise Exception("not implemeted")

# Adapted from `predict` method of `BaseModel` 
# https://github.com/jdb78/pytorch-forecasting/blob/7e376020871b330ac7c92ca4ff68ad430adee203/pytorch_forecasting/models/base_model.py#L1005
def predict_embeddings(
    model,
    data,
    return_index: bool = False,
    return_decoder_lengths: bool = False,
    batch_size: int = 64,
    num_workers: int = 0,
    fast_dev_run: bool = False,
    show_progress_bar: bool = False,
    return_x: bool = False,
    **kwargs,
    ):
    """
    Extract embeddings from the model given input data.
    The function will call `embedding_tensor(model, x, **kwargs)`, make sure it is defined
    for your specific model class.

    Args:
        data: dataloader, dataframe or dataset
        return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
            dataframe corresponds to the first dimension of the output and the given time index is the time index
            of the first prediction)
        return_decoder_lengths: if to return decoder_lengths (in the same order as the output)
        batch_size: batch size for dataloader - only used if data is not a dataloader is passed
        num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
        fast_dev_run: if to only return results of first batch
        show_progress_bar: if to show progress bar. Defaults to False.
        return_x: if to return network inputs (in the same order as prediction output)
        **kwargs: additional arguments to network's forward method

    Returns:
        output, x, index, decoder_lengths: some elements might not be present depending on what is configured
            to be returned
    """
    # convert to dataloader
    if isinstance(data, pd.DataFrame):
        data = TimeSeriesDataSet.from_parameters(model.dataset_parameters, data, predict=True)
    if isinstance(data, TimeSeriesDataSet):
        dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
    else:
        dataloader = data

    # ensure passed dataloader is correct
    assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

    # prepare model
    model.eval()  # no dropout, etc. no gradients

    # run predictions
    output = []
    decode_lenghts = []
    x_list = []
    index = []
    progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
    with torch.no_grad():
        for x, _ in dataloader:
            # move data to appropriate device
            data_device = x["encoder_cont"].device
            if data_device != model.device:
                x = move_to_device(x, model.device)

            ## CALL embeddings_tensor instead of forward
            # out = model(x, **kwargs)  # raw output is dictionary
            out = embedding_tensor(model, x, **kwargs)

            out = move_to_device(out, device="cpu")
            output.append(out)
            if return_x:
                x = move_to_device(x, "cpu")
                x_list.append(x)
            if return_index:
                index.append(dataloader.dataset.x_to_index(x))
            progress_bar.update()
            if fast_dev_run:
                break
    

    # concatenate output (of different batches)
    output = _torch_cat_na(output)
    
    # generate output
    if return_x or return_index or return_decoder_lengths:
        output = [output]
    if return_x:
        output.append(_concatenate_output(x_list))
    if return_index:
        output.append(pd.concat(index, axis=0, ignore_index=True))
    if return_decoder_lengths:
        output.append(torch.cat(decode_lenghts, dim=0))
    return output

def create_model(dataset, params):
    """
    Create model from dataset and params dictionary.
    """
    model_type = params["model"]["type"]
    loss_type = params["train"].get("loss", None)
    if loss_type is not None:
        if loss_type.startswith("mae"):
            loss_lambda = lambda: MAE()
        elif loss_type.startswith("quantile"):
            quantiles = [float(x) for x in loss_type.split('_')[1:]] 
            loss_lambda = lambda: QuantileLoss(quantiles)
        else:
            raise(Exception(f"loss {loss_type} not supported"))
        
    is_multi_target = True if isinstance(params["features"]["targets"], list) else False
    
    if model_type == "TFT":
        if loss_type is None:
            loss_lambda = lambda: QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])  
        if is_multi_target:
            loss = MultiLoss([loss_lambda() for _ in params["features"]["targets"]])
        else:
            loss = loss_lambda()
            
        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=params["train"]["learning_rate"],
            hidden_size=params["model"]["hidden_size"],
            lstm_layers=params["model"]["rnn_layers"],
            attention_head_size=params["model"]["heads"],
            dropout=params["train"]["dropout"],
            hidden_continuous_size=params["model"]["hidden_continuous_size"],
            logging_metrics=[MASE(), MAE(), MAPE(), RMSE(), SMAPE()],
            optimizer = params["train"]["optimizer"],
            loss=loss,
            )

    elif model_type in ["LSTM", "GRU"]:
        if loss_type is None:
            loss_lambda = lambda: MAE()
        if is_multi_target:
            loss = MultiLoss([loss_lambda() for _ in params["features"]["targets"]])
        else:
            loss = loss_lambda()
        
        model = RecurrentNetwork.from_dataset(
                dataset,
                cell_type = model_type,
                learning_rate=params["train"]["learning_rate"],
                hidden_size=params["model"]["hidden_size"],
                rnn_layers=params["model"]["rnn_layers"],
                dropout=params["train"]["dropout"],
                optimizer = params["train"]["optimizer"],
                loss=loss,
                )

    elif model_type == "DeepAR":
        if loss_type is None:
            loss_lambda = lambda: NormalDistributionLoss()
        if is_multi_target:
            loss = MultiLoss([loss_lambda() for _ in params["features"]["targets"]])
        else:
            loss = loss_lambda()
        
        model = DeepAR.from_dataset(
                dataset,
                learning_rate=params["train"]["learning_rate"],
                hidden_size=params["model"]["hidden_size"],
                rnn_layers=params["model"]["rnn_layers"],
                dropout=params["train"]["dropout"],
                optimizer = params["train"]["optimizer"],
                loss=NormalDistributionLoss(),
        )
    else:
        raise Exception(f"Model type {model_type} not supported")
    return model


