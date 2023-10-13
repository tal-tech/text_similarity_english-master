import preprocess as pre
import eval 
import json
def find_text(text,target):
    pre.stopwordslist("./bin/stopwords.txt")

    nn = eval.simEngCheck()
    max_sim = 0
    res  = " "
    for value in text.values():
        result = nn.forward(value,target)
        result_dict = json.loads(result)

        # 现在可以提取 similarity 的值了
        similarity = result_dict["similarity"]
        if similarity > max_sim:
            max_sim = similarity
            res = value
    print(res)
if __name__ == "__main__":
    text = {
        "text1" : "I enjoy listening to music in my free time.",
        "text2" : "Listening to music is something I enjoy doing in my spare time.",
    }
    target="I find pleasure in immersing myself in music during my leisure hours."
    find_text(text,target)

"""
{
    "text1": "I enjoy listening to music in my free time.", 
    "text2": "Listening to music is something I enjoy doing in my spare time.",
    "similarity": 0.6645563244819641
}
"""
