# Лекции и семинары по курсу эффективные модели глубокого обучения

## Лекции

- **Мотивация к курсу и что такое вычислительная эффективность. От чего зависит эффективность работы моделей**   [YouTube](https://youtu.be/jODQiHnZpW8?si=8hCaywKwiXtpcqYw)

- **Автоматический поиск архитектур**  [YouTube](https://youtu.be/YdOcl9QJWrw?si=uf9GXIrLAAnI2hu7)

- **Аппаратное обеспечение и низкоуровневые решения. Введение в вычислительные устройства, как работают CPU и GPU память** [YouTube](https://youtu.be/ul8Fg4PaDwI?si=37Pyhd4Q6hqTCXt5)

- **JIT and Compile в PyTorch 2.0. Погружение в JIT trace, jit compile и Compile**  [YouTube](https://youtu.be/3a4Af-ygcEg?si=tpZ39Er680V3WMa0)

- **Прунинг и спарсификация. Обзор основных методов спарсификации моделей, мотивация почему это работает и какие типы спарсификации бывают** [YouTube](https://youtu.be/K0Ow-Bh58S8?si=ln5hQEcTCe8SeZAN)

- **Квантизация. Основные методы и подходы к квантизации**  [YouTube](https://youtu.be/TyJkig2uR3c?si=y9hd-GmwRXqTAwQK)

- **Продвинутый прунинг — обучение и другое**  [YouTube](https://youtu.be/TFtgFZvMl1w?si=_fT_THC5zTMG_CiO)

- **Методы оптимизации LLM: больших языковых моделей — KV-Cache, Paged Attention, GradientCheckpointing и далее**  [YouTube](https://youtu.be/YVsVv4Z2ZjQ?si=l0gZg_2HueJsOBnZ)

- **Обзор методов поиска архитектур и оптимизации с помощью LLM**  [YouTube](https://youtu.be/wflWNpYPm48?si=WsZPy7TS_U-TRf3r)

## Семинары

- **Профилировка моделей с PyTorch Profiler**  [YouTube](https://youtu.be/wXum5mX-nUE?si=enU35KZUnMXM0FcS)

- **Подбор гиперпараметров для моделей + TPE**  [YouTube](https://youtu.be/TabOL3kQsHo?si=HfS__MFVf2nP9hSq)

- **Введение в Triton. Примеры программирования кернелей на Triton**  [YouTube](https://youtu.be/8RW3aYU41zA?si=x397ipq4pJFLZeBi)

- **Triton + Quant**  [YouTube](https://youtu.be/Um--GUp8b3U?si=Wh48oJX4-VCJrZkb)

- **Работа с JIT, конвертация модели в ONNX. Ускорение модели с JIT и compile**  [YouTube](https://youtu.be/RZoUr0Z1eZk?si=E2hslgdl5grEB32T)

- **Структурный и неструктурный прунинг для VGG**  [YouTube](https://youtu.be/OLHtH78wXYo?si=TcqR9vhD29nQBtGG)

- **Реализация квантования с LSQ. Квантование с PyTorch на CPU**  [YouTube](https://youtu.be/m18jFsQv8Kg?si=FpAOmctzMmJ0iK2A)

- **Продвинутый прунинг**  [YouTube](https://youtu.be/XRyV1IvCSbw?si=26fJ235faSjq2lC3)


# Проекты кураса по категориям:

> Квантование моделей и Triton-оптимизация  
Ускорение и профилирование ASR / Whisper  
Профилирование и энергопотребление  
Квантование с обучением (QAT) и оценка на CPU  
Запуск моделей на мобильных устройствах  
Бинарные нейронные сети  
Поиск архитектур с помощью LLM (NAS)  
Эффективное обучение (Data-Efficient Training)  
Оптимизация свёрточных слоёв

## Квантование моделей и Triton-оптимизация

- **Реализация Triton-кернелей для квантизации весов в LLM и инференса квантизованной модели**
  
(Team 1) [Презентация](https://docs.google.com/presentation/d/1wVt59BXlpecdIsPMsVrmu5LWq643lpo4KoSyTwQw2WY/edit?usp=sharing) | [GitHub](https://github.com/echanatwell/LLM_weight_quantization_triton)

(Team 2)  [Презентация](https://docs.google.com/presentation/d/1EOnwKNEroBrkImiSahOT7TQaaaZoFN6yECTIENl_Zz0/edit?usp=sharing) | [GitHub](https://github.com/LadaNikitina/EDLM_Ledneva)

(Team 3)  [Презентация](https://docs.google.com/presentation/d/1gHym7QXVN8_Eeu4KGZlcihIEFXk__xSdUHJ6kdLeu4M/edit?usp=sharing) | [GitHub](https://github.com/3LayerPerceptron/EDLM-quantization)

(Team 4)  [GitHub](https://github.com/Artyom1363/EDLM-LLMint4quant)

(Team 5)  [Презентация](https://docs.google.com/presentation/d/1jI7AaI7eOdfkCbPZx7HPDU_FzpYwCnEChPuZwRdznY8/edit?usp=sharing) | [GitHub](https://github.com/Maxon081102/int4_bf16)

(Team 6)  [Презентация](https://docs.google.com/presentation/d/1WqyRSKxBo0hWmQvjSuGA0CmQ9lx11_9-LJoqfcvgoYc/edit?usp=sharing) | [GitHub](https://github.com/PeMikj/int4)

(Team 7)  [Презентация](https://docs.google.com/presentation/d/1q5U9GySLYhBtQ6d-rHM0eitG6wglxUJGLER8AKyG--c/edit?usp=sharing) | [GitHub](https://github.com/Mihail-Olegovich/Efficient-DL-Llama-3-Quant)

(Team 8)  [Презентация](https://www.figma.com/slides/uvcAQHNwRq5f87Ti94zanV/int4-triton-kernels-slides?node-id=1-521&t=t1Pqr1ZRwG0lZiSw-1) | [GitHub](https://github.com/sashaboriskin/int4x4-to-fp16-linear)

(Team 9)  [GitHub](https://github.com/ankkarp/TritonQuantLLama)

(Team 10)  [GitHub](https://github.com/anaaaiva/llm-quantization)

## Ускорение и профилирование ASR / Whisper

- **Ускорение ASR (Whisper) моделей для работы в реальном времени**  
  [Презентация](https://github.com/Qeshtir/EDLM_Whisper_Boost/tree/main/presentation) | [GitHub](https://github.com/Qeshtir/EDLM_Whisper_Boost)

- **Ускорение Whisper. Real Time инференс на CPU**  
  [GitHub](https://github.com/fropych/edl-proj)

- **Ускорение ASR (Whisper) моделей для работы в реальном времени**  
  [Презентация](https://docs.google.com/presentation/d/1_tq8MKAajpIV3EdTsrDI5ZAkUaMDo7D2Ksm7vjUtjo0/edit?usp=sharing) | [GitHub](https://github.com/shockless/realtime-whisper)

- **Ускорение ASR (Whisper) моделей для работы в реальном времени**  
  [Презентация](https://docs.google.com/presentation/d/1HsNGIltMY7nRWN2zitW2CxM1pCB-WFqeAgY1cuMuiD8/edit?usp=sharing) | [GitHub](https://github.com/KOTOBOPOT/CPU-Whisper-Benchmark)

- **Профилирование ASR**  
  [Презентация](https://docs.google.com/presentation/d/1bcbs9oQpPnkYEYVdoNrLD8k2BXIDME6asolyl4Yf1D4/edit?usp=sharing) | [GitHub](https://github.com/emildenikaev/EDLM_ASR_Profiling)

## Профилирование и энергопотребление

- **Профилировка VLM**  
  [Презентация](https://docs.google.com/presentation/d/1zxSfHWGDDCnX7AT06WCThN3kelzeoNOS7C89wHEyl08/edit?usp=sharing) | [GitHub](https://github.com/matweykai/vlm-profiling)

- **Профилировка LLM и VLLM (+ потребление энергии)**  
  [Презентация](https://docs.google.com/presentation/d/1HR68rODAJk9CXDNW2O-OZQohVX8OqVOBPVP7vEUzRHo/edit?usp=sharing) | [GitHub](https://github.com/tambovtsev-io/edlm-vlm-profiling)

- **Бенчмарк железа для Local AI**  
  [Figma](https://www.figma.com/slides/dwzpcFvDvaGNs5lGDKhWS8/Laperf-small?node-id=1-117&t=Ty4LXfYwgvzPYLO9-1) • [Google Slides](https://docs.google.com/presentation/d/1xSoKXmm0xTl4_-VDQBwTeNZ9x3vEIR7eo_XnmhuAf6s/edit?usp=sharing) | [GitHub](https://github.com/bogdanminko/laperf) | [Документация](https://bogdanminko.github.io/laperf/)

## Квантование с обучением (QAT) и оценка на CPU

- **QAT с конвертацией в Int8 и оценка качества на CPU**  
  [Презентация](https://docs.google.com/presentation/d/1cdyGmZV9pcd29rgzwK22FsySIkyiwyEFevc8XlCzIqA/edit?usp=sharing) | [GitHub](https://github.com/triflt/eff-dl-project)

- **QAT с конвертацией в Int8 и оценка качества на CPU**  
  [Презентация](https://docs.google.com/presentation/d/1Oc92nSjYlcrlD4KkRq8MpWjE_TiBQTFWomQotJkvN_Y/edit) |  
  [GitHub 1](https://github.com/sadevans/QAT-ESPCN-SASREC-LSTM) •  
  [GitHub 2](https://github.com/BulatMaratovich/FNO/blob/main/FNO-QAT/README.md) •  
  [GitHub 3](https://github.com/BulatMaratovich/LSTM-QAT)

- **QAT с конвертацией в Int8 и оценка качества на CPU**  
  [Презентация](https://docs.google.com/presentation/d/1Y_C-_-7d1jB4C8_2kN98ON75GF8gQcC8erteTHWST7k/edit?usp=sharing) | [GitHub](https://github.com/7Askar7/EDLM-ITMO)

- **QAT с конвертацией в Int8 и оценка качества на CPU**  
  [Презентация](https://docs.google.com/presentation/d/12_qBcHvJq-hn6Eo2TYvHFvkG_Zsm6nz_gUBxCeGfkBo/edit?slide=id.g3a8393c1102_0_9) | [GitHub](https://github.com/tony-pitchblack/qat-eval)

- **Бенчмарк оптимизаторов для QAT**  
  [Презентация](https://docs.google.com/presentation/d/1e16X0hE98xe3QmNbdvs-qM8I3Fjmx4eo7YUTiw_xM7Q/edit?usp=sharing) | [GitHub](https://github.com/import-Julik/QAT_Optimizer_Benchmark)

## Запуск моделей на мобильных устройствах

- **Запуск квантизованной Int8 CartoonGAN-tiny на телефоне**  
  [Презентация](https://docs.google.com/presentation/d/14-auzCPN7U77VEtgCaCorK61ZiJkNn5_z9oz3EQOxlU/edit?usp=sharing) | [GitHub](https://github.com/SuSFCTV/cartoon-gan-optimized/tree/main)

- **Запуск квантизованной Int8 CartoonGAN-tiny или TTS на телефоне**  
  [Презентация](https://docs.google.com/presentation/d/1MLsNklbNL4z6cITJRt1NdsEiCF2jCMqO1IPZq1nLm74/edit?usp=sharing) | [GitHub](https://github.com/Dforgeek/phone-tts)

- **Ускорение модели Cartoon GAN**  
  [Презентация](https://docs.google.com/presentation/d/15v1i8I5W0sZMK0ze4y-cvD81gsSKkWTRWHbNlvGqYss/edit?slide=id.g3a8423dc999_1_65) | [GitHub](https://github.com/OrAnge-Lime/ITMO_EDLM_project)

## Бинарные нейронные сети

- **Обучение модели с бинарными весами для задачи классификации голосовых команд**  
  [Презентация](https://github.com/OSurD/BinaryNet/blob/master/binaryNet.pdf) | [GitHub](https://github.com/OSurD/BinaryNet)

- **Обучение модели с бинарными весами для задачи классификации голосовых команд**  
  [GitHub](https://github.com/fake-or-lake/binary-kws)

## Поиск архитектур с помощью LLM (NAS)

- **Автоматический поиск архитектуры под простую задачу с помощью LLM**
  
(Team 1) [GitHub](https://github.com/ITMO-EDLM-TEAM/LLM-NAS/tree/develop)


(Team 2)  [GitHub](https://github.com/Irlirion/NAS-LLM-Agent)


(Team 3)  [GitHub](https://github.com/WpythonW/NAS_LLM)


(Team 4)  [Презентация](https://docs.google.com/presentation/d/1Ooo3s7lbtlGgAGiRQmD4AQlmL-80FjXKzvKpgW8KJH4/edit?usp=sharing) | [GitHub](https://github.com/dzaripov/arch_agent)


(Team 5)  [Презентация](https://docs.google.com/presentation/d/1kKUjACGS7QuwrOKG37Lat73nlz80iMsur8bKHzFLGXY/edit?usp=sharing) | [GitHub](https://github.com/slewie/LLMNas)


(Team 6)  [Презентация](https://docs.google.com/presentation/d/1NjTwp3XMCbySExYtCGc1a0AKN-4IFs9z8BTklwbuWAI/edit?usp=sharing) | [GitHub](https://github.com/Fourzeroo/Efficient-model)

## Эффективное обучение (Data-Efficient Training)

- **Data-Efficient Training**  
  [Презентация](https://docs.google.com/presentation/d/1WH8Rehl0b8xZYDhjvW41tIw_-lu6J7JcqxJvmUXBJgc/edit?usp=sharing) | [GitHub](https://github.com/IsachenkoBogdan/llm-data-selection-bert)

- **Data-Efficient Training**  
  [Презентация](https://docs.google.com/presentation/d/1uBuXALY-gL28Q0zb4bIBhyf6BZderT-Bjen2o7VacJg/edit?usp=sharing) | [GitHub](https://github.com/Bovrrr/itmo-edlm-course-project-train-data-sampling)

- **Data-Efficient Training — Knowledge Distillation от большой к tiny модели на маленьком 10% данных**  
  [Презентация](https://docs.google.com/presentation/d/1ezU5i-dlzWrnJaix4gswlcNKfJifOUsj3TPGTdwMOnY/edit?usp=sharing) | [GitHub](https://github.com/ITMO-EDLM/data-efficiency)

## Оптимизация свёрточных слоёв

- **Conv2d Reimagined: img2col, GEMM, Sparsity & Quantization**  
(Team 1)  [Презентация](https://docs.google.com/presentation/d/1gf_vmIJENtD3EbacVWSf3uub7m1ReNeb0w1AEaTqNcQ/edit?usp=sharing) | [GitHub](https://github.com/GlebIsrailevich/edlm-conv2d)

(Team 2)  [Презентация](https://docs.google.com/presentation/d/1VOi-fSxuikIaxMdJ_PmxwPcwdqYGmtbZd0JjFSfKVp8/edit?slide=id.p) | [GitHub](https://github.com/EugenePWN/EDLM_Project_ITMO/tree/main)

(Team 3)  [Презентация](https://docs.google.com/presentation/d/12Ua0Q5049UrwVbi2t9F-IiIduJWs7aDHlWfTBjQ7KGk/edit?slide=id.g3a4586df2d4_0_252) | [GitHub](https://github.com/kolyan288/conv2d_reimagined/tree/main)

(Team 4)  [Презентация](https://docs.google.com/presentation/d/16KEv39tyYu8aMYDQZiRNKGKFPUUWUEVm5UPZs_67Lps/edit?slide=id.g3a83eddbab8_0_5) | [GitHub](https://github.com/Paradocsal/Cond2dReimagined)


(Team 5)  [Презентация](https://docs.google.com/presentation/d/1JExJh_En-gjypy2seUJGrNnSgyFWH41u3AbuARI7a9Q/edit?slide=id.g3a8109c197c_2_0) | [GitHub](https://github.com/manzhura/conv2d-img2col-gemm)
