

# 微软认知服务-语音部分



## 1. 认知服务基本概念

### 1.1 认知服务-是什么

认知服务使每个开发人员和数据科学家都能接触到 AI。借助领先模型，可以解锁各种用例。只需进行 API 调用，即可将查看、收听、朗读、搜索、理解和加速高级决策的功能嵌入到应用中。让所有技能级别的开发人员和数据科学家都能够轻松地向其应用添加 AI 功能。

Azure 认知服务是具有 REST API 和客户端库 SDK 的基于云的服务，可用于帮助你将认知智能构建到应用程序中。 即使你没有人工智能 (AI) 或数据科学技能，也可向应用程序添加认知功能。 Azure 认知服务包含各种 AI 服务，让你能够构建可以看、听、说、理解，甚至可以决策的认知解决方案。

一句话，我们可以通过SDK或API直接调用微软提供的，已经训练好的AI模型（影像，语音，语言，决策等），把这种AI功能增加我们的解决方案里。



[官方宣传片](https://azure.microsoft.com/zh-cn/services/cognitive-services/#overview)





### 1.2 认知服务-有哪些分类

- 影像： 分析图形和视频内容、人脸识别

- 语音： TTS，STT 、语音翻译等

- 语言：实体识别，意图识别，情感分析

- 决策：个性化服务，异常检测器、内容审查器等

  

<img src="cognitive.png" alt="image-20220117141442420" style="zoom: 80%;" />



语音都提供哪些功能

<img src="C:\work\AITech\yuyin.png" style="zoom:80%;" />

### 1.3 认知服务-如何使用

- Azure 云上创建认知服务，提供subscription，供外部直接调用
- 提供多个 Docker 容器，可以私有话部署





## 2. 认知服务SDK安装



使用认知服务，在python环境下，需要新安装服务包

```python
pip install azure-cognitiveservices-speech 
```



## 3. TTS 文字转语音

TTS (Text to Speech) ，语音转文字，语音合成功能

使用 119 种语言和变体，超过 270 种神经语音来吸引全球观众。使用极具表现力和类似人类的声音将你的方案(如文本阅读器和支持语音的助手)变为现实。神经文本到语音转换功能支持若干种话语风格，包括聊天、新闻播报和客户服务，以及各种情感(如快乐和同情)。



**多种调用方式**

1. 调用SDK方式

2. REST API 方式

   ...



### 3.1 使用语音合成服务

1）普通文本的语音合成

```python
import azure.cognitiveservices.speech as speechsdk

#配置信息
speech_key, speech_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

#构造语音合成服务
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

#文字转语音
text="文字描述部分"
result = speech_synthesizer.speak_text_async(text).get()
```



对合成结果的一个判断

```python
 if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("识别结果 [{}]".format(text))
 if result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("识别取消: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("错误: {}".format(cancellation_details.error_details))
```



### 3.2 一些配置可选项

#### 3.2.1 语言的配置

```python
language = "zh-CN"  #default en-US
speech_key, speech_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

#转换成语音时候，选择需要转换的语言
speech_config.speech_synthesis_language = language
```

目前微软认知服务支持119种语言，具体参考 [支持语言list]( https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/language-support#text-to-speech)



#### 3.2.2 语音的配置

对于不同的语言会存在不通过种类的voice 供调用者使用，可以根据需求任意切换自己需要的声音

```python
speech_key, speech_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

#voice配置
voice = "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)"
speech_config.speech_synthesis_voice_name = voice
```

全部支持语音参考：https://aka.ms/csspeech/voicenames

中文支持如下几种

| 中文（粤语，繁体）   | `zh-HK` | Female | `zh-HK-HiuGaaiNeural`   | 常规                                                         |
| -------------------- | ------- | ------ | ----------------------- | ------------------------------------------------------------ |
| 中文（粤语，繁体）   | `zh-HK` | Female | `zh-HK-HiuMaanNeural`   | 常规                                                         |
| 中文(粤语，繁体)     | `zh-HK` | 男     | `zh-HK-WanLungNeural`   | 常规                                                         |
| 中文（普通话，简体） | `zh-CN` | 女     | `zh-CN-XiaohanNeural`   | 常规，[使用 SSML](https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/speech-synthesis-markup#adjust-speaking-styles) 提供多种风格 |
| 中文（普通话，简体） | `zh-CN` | 女     | `zh-CN-XiaomoNeural`    | 常规，[使用 SSML](https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/speech-synthesis-markup#adjust-speaking-styles) 提供多种角色扮演和风格 |
| 中文（普通话，简体） | `zh-CN` | 女     | `zh-CN-XiaoruiNeural`   | 高级语音，[使用 SSML](https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/speech-synthesis-markup#adjust-speaking-styles) 提供多种风格 |
| 中文（普通话，简体） | `zh-CN` | 女     | `zh-CN-XiaoxiaoNeural`  | 常规，[使用 SSML](https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/speech-synthesis-markup#adjust-speaking-styles) 提供多种语音风格 |
| 中文（普通话，简体） | `zh-CN` | 女     | `zh-CN-XiaoxuanNeural`  | 常规，[使用 SSML](https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/speech-synthesis-markup#adjust-speaking-styles) 提供多种角色扮演和风格 |
| 中文（普通话，简体） | `zh-CN` | 女     | `zh-CN-XiaoyouNeural`   | 儿童语音，针对讲故事进行了优化                               |
| 中文（普通话，简体） | `zh-CN` | 男     | `zh-CN-YunxiNeural`     | 常规，[使用 SSML](https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/speech-synthesis-markup#adjust-speaking-styles) 提供多种风格 |
| 中文（普通话，简体） | `zh-CN` | 男     | `zh-CN-YunyangNeural`   | 针对新闻阅读进行了优化， [使用 SSML](https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/speech-synthesis-markup#adjust-speaking-styles) 提供多种语音风格 |
| 中文（普通话，简体） | `zh-CN` | 男     | `zh-CN-YunyeNeural`     | 针对讲故事进行了优化                                         |
| 中文(台湾普通话)     | `zh-TW` | Female | `zh-TW-HsiaoChenNeural` | 常规                                                         |
| 中文(台湾普通话)     | `zh-TW` | Female | `zh-TW-HsiaoYuNeural`   | 常规                                                         |
| 中文(台湾普通话)     | `zh-TW` | 男     | `zh-TW-YunJheNeural`    | 常规                                                         |



另外，如果官方提供的声音都不能满足，你可以自己定义声音的

详细参看：https://aka.ms/customvoice

```python
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.endpoint_id = "YourEndpointId"
speech_config.speech_synthesis_voice_name = "YourVoiceName"
```





### 3.3 实战例子

#### 例子1-文字转语音扬声器播放

```python
import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"
language = "zh-CN"

def tts_to_speaker():
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_language = language
    voice = "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaohanNeural)"
    speech_config.speech_synthesis_voice_name = voice

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    while True:
        print("输入文字转换为语音, Ctrl-Z 退出")
        try:
            text = input()
        except EOFError:
            break
        result = speech_synthesizer.speak_text_async(text).get()

        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("转换语音的文字 [{}]".format(text))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("语音合成取消: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("错误: {}".format(cancellation_details.error_details))

tts_to_speaker()
```



#### 例子2- 文字转语音保存成文件



保存文件只需要在合成语音服务

```python
file_name = "outputaudio1.wav"
file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
```

完成代码

```python
import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"
language = "zh-CN"

def tts_to_file():
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_language=language

    file_name = "outputaudio1.wav"
    file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)

    while True:
        print("输入文字转换为语音, Ctrl-Z 退出")
        try:
            text = input()
        except EOFError:
            break
        result = speech_synthesizer.speak_text_async(text).get()
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("转换语音的文字 [{}]".format(text))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("语音合成取消: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("错误: {}".format(cancellation_details.error_details))

tts_to_file()
```





### 3.4 SSML（语音合成标记语言）使用

这里是一些高级选项，需要使用SSML来转换语音了，SSML结构如下

```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
 ...
</speak>
```

最基本的一个SSML结构体，lang用来指定语言



#### 3.4.1 语音 voice

voice参数 的name就是用来制定讲话语音的，简单列举一些常用的

- `zh-CN-XiaohanNeural`
- `zh-CN-XiaomoNeural`
- `zh-CN-XiaoxuanNeural`
- `zh-CN-XiaoruiNeural`
- `en-US-AriaNeural`
- `en-US-JennyNeural`

使用语音合成标记语言SSML

```python
#SSML文字转语音
txt = "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" xml:lang=\"zh-CN\">"
txt += " <voice name=\"zh-CN-XiaomoNeural\">"
txt += "   你好，欢迎是用微软认知服务"
txt += " </voice>"
txt += " <voice name=\"zh-CN-YunxiNeural\">"
txt += "   你好，欢迎是用微软认知服务"
txt += " </voice>"
txt += "</speak>"
result = speech_synthesizer.speak_ssml_async(txt).get()
```





####     3.4.2 讲话风格配置 -express-as

为了更进一步描述讲话风格，微软定义了新标记mstts:express-as，可以用来更精确的表达语言的轻重、甚至扮演不同年龄的角色等。

```xml
<voice name="zh-CN-XiaomoNeural">
    <mstts:express-as style="cheerful" role="Girl"  styledegree="1">
        你好，欢迎您使用微软认知服务
    </mstts:express-as>
</voice> 
```



注意使用express-as标签需要引入命名空间 xmlns:mstts

```xml
<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\"  xmlns:mstts=\"https://www.w3.org/2001/mstts\">
   ...
</speak>
```



| 属性          | 说明                                                         | 必需/可选                                                    |
| :------------ | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `style`       | 指定讲话风格。 目前，讲话风格特定于语音。                    | 如果调整神经语音的讲话风格，则此属性是必需的。 如果使用 `mstts:express-as`，则必须提供风格。 如果提供无效的值，将忽略此元素。 |
| `styledegree` | 指定说话风格的强度。 接受的值：0.01 到 2（含边界值）。 默认值为 1，表示预定义的风格强度。 最小单位为 0.01，表示略倾向于目标风格。 值为 2 表示是默认风格强度的两倍。 | 可选（目前，`styledegree` 仅支持中文（普通话，简体）神经语音。） |
| `role`        | 指定讲话角色扮演。 语音将充当不同的年龄和性别，但语音名称不会更改。 | 可选（`role` 仅支持 zh-CN-XiaomoNeural 和 zh-CN-XiaoxuanNeural。） |



**Style 属性**

| `zh-CN-XiaomoNeural` | `style="calm"`        | 以沉着冷静的态度说话。 语气、音调、韵律与其他语音类型相比要统一得多。 |
| -------------------- | --------------------- | ------------------------------------------------------------ |
|                      | `style="cheerful"`    | 以较高的音调和音量表达欢快、热情的语气                       |
|                      | `style="angry"`       | 以较低的音调、较高的强度和较高的音量来表达恼怒的语气。 说话者处于愤怒、生气和被冒犯的状态。 |
|                      | `style="fearful"`     | 以较高的音调、较高的音量和较快的语速来表达恐惧、紧张的语气。 说话者处于紧张和不安的状态。 |
|                      | `style="disgruntled"` | 表达轻蔑和抱怨的语气。 这种情绪的语音表现出不悦和蔑视。      |
|                      | `style="serious"`     | 表达严肃和命令的语气。 说话者的声音通常比较僵硬，节奏也不那么轻松。 |
|                      | `style="depressed"`   | 调低音调和音量来表达忧郁、沮丧的语气                         |
|                      | `style="gentle"`      | 以较低的音调和音量表达温和、礼貌和愉快的语气                 |



**Role 属性**

| 角色                      | 说明                       |
| :------------------------ | :------------------------- |
| `role="Girl"`             | 该语音模拟女孩。           |
| `role="Boy"`              | 该语音模拟男孩。           |
| `role="YoungAdultFemale"` | 该语音模拟年轻成年女性。   |
| `role="YoungAdultMale"`   | 该语音模拟年轻成年男性。   |
| `role="OlderAdultFemale"` | 该语音模拟年长的成年女性。 |
| `role="OlderAdultMale"`   | 该语音模拟年长的成年男性。 |
| `role="SeniorFemale"`     | 该语音模拟老年女性。       |
| `role="SeniorMale"`       | 该语音模拟老年男性。       |



例子：

```python
txt = "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\""
txt += " xmlns:mstts=\"https://www.w3.org/2001/mstts\" xml:lang=\"zh-CN\">"
txt += "<voice name=\"zh-CN-XiaoxiaoNeural\">"
txt += " <mstts:express-as style=\"sad\" styledegree=\"2\">"
txt += "       快走吧，路上一定要注意安全，早去早回。"
txt += " </mstts:express-as>"
txt += "</voice>"
txt += "</speak>"
result = speech_synthesizer.speak_ssml_async(txt).get()
```





#### 3.4.3 调整语言-lang

SSML结构中使用lang标签

```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="en-US-JennyMultilingualNeural">
        I am looking forward to the exciting things.
        <lang xml:lang="zh-CN">
            你好，欢迎您使用微软认知服务
        </lang>
        <lang xml:lang="en-US">
            Welcome to use azure cognitive services
        </lang>
    </voice>
</speak>
```



例子：

```python
txt = "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\""
txt += "       xmlns:mstts=\"https://www.w3.org/2001/mstts\" xml:lang=\"en-US\">"
txt += "    <voice name=\"en-US-JennyMultilingualNeural\">"
txt += "        I am looking forward to the exciting things."
txt += "        <lang xml:lang=\"zh-CN\">"
txt += "            你好，欢迎您使用微软认知服务"
txt += "        </lang>"
txt += "        <lang xml:lang=\"ja-JP\">"
txt += "            你好，欢迎您使用微软认知服务"
txt += "        </lang>"
txt += "    </voice>"
txt += "</speak>"
result = speech_synthesizer.speak_ssml_async(txt).get()
```



#### 3.4.4 静音处理-silence

标签mstts:silence 用来控制静音的

```
<mstts:silence  type="Sentenceboundary" value="200ms"/>
```

属性使用

| 属性    | 说明                                                         | 必需/可选 |
| :------ | :----------------------------------------------------------- | :-------- |
| `type`  | 指定添加静音的位置：`Leading` - 在文本的开头`Tailing` - 在文本的结尾`Sentenceboundary` - 在相邻句子之间 | 必须      |
| `Value` | 指定暂停的绝对持续时间，以秒或毫秒为单位；该值应设为小于 5000 毫秒。 例如，`2s` 和 `500ms` 是有效值 | 必须      |



具体用例

```python
    txt = "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\""
    txt += "       xmlns:mstts=\"https://www.w3.org/2001/mstts\" xml:lang=\"zh-CN\">"
    txt += "<voice name=\"zh-CN-XiaoxiaoNeural\">"
    txt += "  <mstts:silence  type=\"Sentenceboundary\" value=\"5s\"/>"
    txt += "    我们快点走吧，看天气要下雨了。"
    txt += "    不行，我必须这些东西打包好，"
    txt += "    那好吧，我等你一起"
    txt += "</voice>"
    txt += "</speak>"
    result = speech_synthesizer.speak_ssml_async(txt).get()
```





#### 3.4.5 语速音量的调整-prosody

prosody 用来调整速率，音量，持续时间等声音参数

```xml
<prosody pitch="value" rate="value" ...>
</prosody>
```



**pitch 参数**

指示文本的基线音节。 可将音调表述为

- 以某个数字后接“Hz”（赫兹）表示的绝对值。 例如 `<prosody pitch="600Hz">some text</prosody>`。
- 以前面带有“+”或“-”的数字，后接“Hz”或“st”（用于指定音节的变化量）表示的相对值。 例如 `<prosody pitch="+80Hz">some text</prosody>` 或 `<prosody pitch="-2st">some text</prosody>`。 “st”表示变化单位为半音，即，标准全音阶中的半调（半步）。
- 常量值：
  - x-low
  - low
  - medium
  - high
  - x-high
  - default

**rate 参数**

rate	指示文本的讲出速率。 可将 rate 表述为：

- 以充当默认值倍数的数字表示的相对值。 例如，如果值为 *1*，则速率不会变化。 如果值为 *0.5*，则速率会减慢一半。 如果值为 *3*，则速率为三倍。
- 常量值：
  - x-slow
  - slow
  - medium
  - fast
  - x-fast
  - default



例子：

```python
    txt = "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\""
    txt += "       xmlns:mstts=\"https://www.w3.org/2001/mstts\" xml:lang=\"zh-CN\">"
    txt += "    <voice name=\"zh-CN-XiaoxiaoNeural\">"
    txt += "        女儿看见父亲走了进来，问道："
    txt += "        <mstts:express-as role=\"YoungAdultFemale\" style=\"cheerful\">"
    txt += "            您来的挺快的，怎么过来的？"
    txt += "        </mstts:express-as>"
    txt += "        父亲放下手提包，说："
    txt += "    </voice>"
    txt += "    <voice name=\"zh-CN-YunxiNeural\">"
    txt += "       <prosody rate=\"0.8\" pitch=\"low\">"
    txt += "        <mstts:express-as style=\"sad\" styledegree=\"2\"  >"
    txt += "           刚打车过来的，一路还算顺利。"
    txt += "        </mstts:express-as>"
    txt += "     </prosody>"
    txt += "    </voice>"
    txt += "</speak>"

    result = speech_synthesizer.speak_ssml_async(txt).get()
```



官方提供了一个用来测试的UI Demo，[在线测试](https://azure.microsoft.com/zh-cn/services/cognitive-services/text-to-speech/#features)



#### 课间练习：

每个小组使用SSML语言来编写一段情景对话，并将该对话保存成wave文件



**要求：**

1）尽可能使用我们学习到的标记

2）对话内容控制在10句以内

3）对话内容，自由发挥



## 4. STT 语音转文字

STT(Speech to text) ，文字转语音，



### 4.1 从麦克风转文字

```python
import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"

def stt_from_mic():
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = 'zh-CN'

    speech_recongnizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    result = speech_recongnizer.recognize_once()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

stt_from_mic()
```



### 4.2 从语音文件转文字



```python
import  azure.cognitiveservices.speech as speechsdk

speech_key, service_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"

def stt_from_file():
    speech_config=speechsdk.SpeechConfig(subscription=speech_key,region=service_region)
    speech_config.speech_recognition_language='zh-CN'
    audio_config=speechsdk.audio.AudioConfig(filename="outputaudio1.wav")

    speech_recognizer=speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("识别结果: {}".format(result.text))
    else:
        print("识别失败")

stt_from_file()
```





