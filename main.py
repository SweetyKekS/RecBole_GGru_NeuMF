import class_ggr
from class_ggr import GGR_NeuMF



if __name__ == '__main__':
    website_path = 'csv_file/website.csv'
    ggru = GGR_NeuMF(path=website_path, epochs= 500, pretrain=True)
    # print(ggru.show_prediction(20))
    # ggru.check_result_by_content(666)
    # ggru.to_csv()
    # print(ggru.pred_to_list(666))
    print(ggru.to_dict()[666])
