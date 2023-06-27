# FreqView
音频频谱动效生成  

-- 使用例 --  
`
g++ -Wall -O2 FreqView4.cc -lSDL2_ttf -lSDL2 -lglew32 -s -o FreqView4.exe  
FreqView4.exe xxx.wav  
`

按 ADQEWS上下左右键调整视角（英文输入模式）  
按空格开始渲染  
生成FreqView.mp4  
__需要同目录下有 font.ttf 用于绘制文字__  
__需要环境中有ffmpeg__  
