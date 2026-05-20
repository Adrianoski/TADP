from torch.utils.data import DataLoader


def prepare_dataset(dataset, tokenizer, text_field='text', max_length=512, batch_size=4):
    def tokenize_function(examples):
        if text_field in examples:
            texts = examples[text_field]
        elif 'sms' in examples:
            texts = examples['sms']
        else:
            texts = examples[list(examples.keys())[0]]

        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False)
