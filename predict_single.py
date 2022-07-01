import torch, configargparse
from data import load_asap_data
from model_architechure_bert_multi_scale_multi_loss_single import DocumentBertScoringModel
from random import randrange


def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--bert_model_path', help='bert_model_path')
    p.add('--efl_encode', action='store_true', help='is continue training')
    p.add('--r_dropout', default=0.0, type=float)
    p.add('--batch_size', default=32, type=int)
    p.add('--bert_batch_size', help='bert_batch_size', type=int)
    p.add('--cuda', action='store_true', help='use gpu or not')
    p.add('--device')
    p.add('--model_directory', default='/home/kixlab3/auto-ui/models')
    p.add('--test_file', help='test data file')
    p.add('--data_dir', default='/home/kixlab3/auto-ui/data', help='data directory to store asap experiment data')
    p.add('--data_sample_rate', default=1.0, type=float)
    p.add('--prompt', default='p8')
    p.add('--fold', default=3, type=int)
    p.add('--chunk_sizes', default='90_30_130_10', type=str)
    p.add('--result_file', default='/home/kixlab3/auto-ui/pred.txt', help='pred result file path', type=str)

    args = p.parse_args()
    args.test_file = f"{args.data_dir}/p8_fold3_test.txt"
    args.model_directory = f"{args.model_directory}/{args.prompt}_{args.fold}"
    args.bert_model_path = args.model_directory

    if torch.cuda.is_available() and args.cuda:
        args.device = 'cuda'
    else:
        args.dev = 'cpu'
    return args


if __name__ == "__main__":
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["asap.ini"])
    args = _initialize_arguments(p)
    print(args)

    # load data
    test = load_asap_data(args.test_file)

    test_documents, test_labels = [], []
    for _, text, label in test:
        test_documents.append(text)
        test_labels.append(label)

    rint = randrange(len(test_documents))
    sample_docum = [test_documents[rint]]
    sample_label = [test_labels[rint]]

    print(f"sample document: {sample_docum}")
    print(f"label label: {sample_label}")

    model = DocumentBertScoringModel(args=args)
    pred, answ = model.predict_for_regress((sample_docum, sample_label))

    print(f"predicted result: {pred}, answer: {answ}")
