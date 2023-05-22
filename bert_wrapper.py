def apply_layer(bert_layer, inputs, output_attentions=False):
    '''
    Mimic the behaviour of a Transformer layer in the encoder
    part of BERT while preserving the output of the feed-forward
    sublayer, in addition to the attention weights.
    '''

    # Apply self-attention to the output of the previous layer
    # and combine the result with the residuals.
    self_attention_outputs = bert_layer.attention(
        inputs,
        output_attentions=output_attentions)
    attention_output = self_attention_outputs[0]
    # Attention weights if requested.
    outputs = self_attention_outputs[1:]

    # Project to a higher-dimensinal space and apply the activation.
    intermediate_output = bert_layer.intermediate(attention_output)

    # Project back and apply dropout --- the result is the output of the FF layer.
    # We extract it before applying dropout to reduce noise in the analysis.
    FF_output = bert_layer.output.dense(intermediate_output)
    intermediate_output = bert_layer.output.dropout(FF_output)

    # Add the outputs of the residual connection and apply layer normalisation.
    layer_output = bert_layer.output.LayerNorm(
        intermediate_output + attention_output)

    return {
        'attention_weights': outputs,
        'FF_output': FF_output,
        'layer_output': layer_output
    }


def apply_layer_w_mask(bert_layer, inputs, attention_mask, output_attentions=False):
    '''
    Mimic the behaviour of a Transformer layer in the encoder
    part of BERT while preserving the output of the feed-forward
    sublayer, in addition to the attention weights.
    '''

    # Apply self-attention to the output of the previous layer
    # and combine the result with the residuals.
    self_attention_outputs = bert_layer.attention(
        inputs,
        attention_mask=attention_mask,
        output_attentions=output_attentions)
    attention_output = self_attention_outputs[0]
    # Attention weights if requested.
    outputs = self_attention_outputs[1:]

    # Project to a higher-dimensinal space and apply the activation.
    intermediate_output = bert_layer.intermediate(attention_output)

    # Project back and apply dropout --- the result is the output of the FF layer.
    # We extract it before applying dropout to reduce noise in the analysis.
    FF_output = bert_layer.output.dense(intermediate_output)
    intermediate_output = bert_layer.output.dropout(FF_output)

    # Add the outputs of the residual connection and apply layer normalisation.
    layer_output = bert_layer.output.LayerNorm(
        intermediate_output + attention_output)

    return {
        'attention_weights': outputs,
        'FF_output': FF_output,
        'layer_output': layer_output
    }


def apply_bert_model(bert_model, tokeniser, inputs, output_attentions=False):
    tokeniser_output = tokeniser(
        inputs, return_tensors='pt', truncation=True, padding=True)
    embedding_output = bert_model.embeddings(
        input_ids=tokeniser_output.input_ids,
        token_type_ids=tokeniser_output.token_type_ids)
    outputs = tuple()
    ff_outputs = tuple()
    attentions = tuple()
    for i, layer_module in enumerate(bert_model.encoder.layer):
        if i == 0:
            previous_output_dict = apply_layer(
                layer_module,
                embedding_output,
                output_attentions=output_attentions)
        else:
            previous_output_dict = apply_layer(
                layer_module,
                previous_output_dict['layer_output'],
                output_attentions=output_attentions)
        outputs = outputs + (previous_output_dict['layer_output'],)
        ff_outputs = ff_outputs + (previous_output_dict['FF_output'],)
        attentions = attentions + (previous_output_dict['attention_weights'],)
    if output_attentions:
        return outputs, ff_outputs, attentions
    else:
        return outputs, ff_outputs


def apply_bert_model_to_pretokinsed(bert_model, inputs, output_attentions=False):
    embedding_output = bert_model.embeddings(
        input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'])
    outputs = tuple()
    ff_outputs = tuple()
    attentions = tuple()
    for i, layer_module in enumerate(bert_model.encoder.layer):
        if i == 0:
            previous_output_dict = apply_layer(
                layer_module,
                embedding_output,
                output_attentions=output_attentions)
        else:
            previous_output_dict = apply_layer(
                layer_module,
                previous_output_dict['layer_output'],
                output_attentions=output_attentions)
        outputs = outputs + (previous_output_dict['layer_output'],)
        ff_outputs = ff_outputs + (previous_output_dict['FF_output'],)
        attentions = attentions + (previous_output_dict['attention_weights'],)
    if output_attentions:
        return outputs, ff_outputs, attentions
    else:
        return outputs, ff_outputs
