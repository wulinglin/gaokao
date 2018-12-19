import pandas as pd
bwc_dict={'chong': [('西南大学', 0.6719999999999999), ('西南石油大学', 0.501), ('四川外国语大学', 0.444), ('四川农业大学', 0.371)],
          'bao':[('西南大学', 0.6719999999999999), ('西南石油大学', 0.501), ('四川外国语大学', 0.444), ('四川农业大学', 0.371)]}
answer = '根据您的信息，我们为您推荐的学校有：<br>'
print(answer)
for key, value in bwc_dict.items():
    print(key, value)
    # for each in value:
    if key == 'wen':
        answer_ = '稳：'
        c = 0
        for each in value:
            if c != len(value) - 1:
                answer_ += str(each[0]) + '、'
            else:
                answer_ += str(each[0]) + '；<br>'
            c+=1
    elif key == 'bao':
        answer_ = '保：'
        c = 0
        for each in value:
            if c != len(value) - 1:
                answer_ += str(each[0]) + '、'
            else:
                answer_ += str(each[0]) + '；<br>'
            c+=1
    elif key == 'chong':
        answer_ = '冲：'
        c = 0
        for each in value:
            if c != len(value) - 1:
                answer_ += str(each[0]) + '、'
            else:
                answer_ += str(each[0]) + '；<br>'
            c += 1
    print(answer_)
    print(answer)
    answer += answer_
print('hhhhhh')
print(answer)