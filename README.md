# IDEAL
Github repo for the method Inconsistency-based virtual aDvErsarial Active Learning (IDEAL).

## Dataset
We now provide the datasets we used in our method IDEAL, i.e., the Legal Text dataset and the Bidding dataset. The Legal Text dataset is collected from the website https://wenshu.court.gov.cn/. The Bidding dataset is collected from the website https://www.cecbid.org.cn/. Their annotations need experts with domain knowledge. The Legal Text dataset is used to classify and retrieve similar cases. We define 12 fact labels (elements of judgment basis) for each case. The Bidding dataset is used to facilitate the users to filter the procurement methods (different companies prefer different procurement methods) and help them locate the most suitable bidding opportunities. We define 22 labels of purchase and sale for each announcement document. We only provide the vectorial representations of labels due to the copyright of these datasets, i.e., the desensitized version. The textual data and annotations of the Legal Text dataset (**legal_text.json**) and the Bidding dataset (**bidding_text.json**) are in the **industrial datasets** folder.

Due to the data-release regularities of Alibaba, we only provide 3000 samples for each dataset (10% of each dataset).

## Code
Due to a tight regularition, the source code of IDEAL is not ready for a clean release right now. We pick up part of the code for open source.
