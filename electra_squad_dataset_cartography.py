from transformers.models.electra import ElectraForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from torch.nn import CrossEntropyLoss
from collections import defaultdict
import torch



# Copied and modified from https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/electra/modeling_electra.py#L1322
def MakeElectraSQUADCartography():
    row_idx_to_loss = defaultdict(lambda: {"loss": [], "last": None})
        
    def forward(
        self: ElectraForQuestionAnswering,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        row_idx=None
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if row_idx != None:
                assert len(row_idx) == len(total_loss)
                for i in range(len(row_idx)):
                    r = int(row_idx[i])
                    row_idx_to_loss[r]["loss"].append(float(total_loss[i].detach()))
                    
                    if row_idx_to_loss[r]["last"] != None:
                        assert input_ids.shape[0] == len(row_idx)
                        last = row_idx_to_loss[r]["last"]
                        assert torch.equal(last, input_ids[i]), f"{last} != {input_ids[i]}, {input_ids[i].shape, last.shape}"
                    else:
                        row_idx_to_loss[r]["last"] = input_ids[i].detach()

            total_loss = total_loss.mean()


        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    return row_idx_to_loss, forward