# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.4: Matrix Multiplication
![Matrix Multiplication](image/Matrix_Multiplication.png)

### Timing Summary
| **Matrix Size** | **Fast (CPU) Time (s)** | **GPU Time (s)** |
|------------------|-------------------------|------------------|
| 64               | 0.00301                | 0.00581          |
| 128              | 0.01602                | 0.01260          |
| 256              | 0.11013                | 0.05844          |
| 512              | 1.26453                | 0.22964          |
| 1024             | 8.09818                | 0.96199          |

### Observations
- **Fast (CPU)** performs better for smaller matrix sizes (e.g., 64).
- **GPU** demonstrates significant speedup for larger matrix sizes (e.g., 1024).

## Task 3.5: Training
### Classification of Simple Model

**CPU:**
- `Hidden dimensions`: 100
- `Learning rate`: 0.05
- `Number of epochs`: 500

Here, time per epoch is 0.034s.
```bash
Epoch  0  loss  5.416418669394428 correct 40
Epoch  10  loss  2.624766066004023 correct 47
Epoch  20  loss  2.4529293909551595 correct 50
Epoch  30  loss  0.5175420926984176 correct 50
Epoch  40  loss  0.29456170440428664 correct 50
Epoch  50  loss  0.44225297910741784 correct 50
Epoch  60  loss  0.9765907492430895 correct 50
Epoch  70  loss  0.8687542813466181 correct 50
Epoch  80  loss  1.1279532780893362 correct 50
Epoch  90  loss  0.05004393067535251 correct 50
Epoch  100  loss  0.0897954051870814 correct 50
Epoch  110  loss  0.5117281943524459 correct 50
Epoch  120  loss  0.13764302836535652 correct 50
Epoch  130  loss  0.19077351900717077 correct 50
Epoch  140  loss  0.4189919551429057 correct 50
Epoch  150  loss  0.20412399181413024 correct 50
Epoch  160  loss  0.09433953780089151 correct 50
Epoch  170  loss  0.026294141332545304 correct 50
Epoch  180  loss  0.23658488759263618 correct 50
Epoch  190  loss  0.10357610023513897 correct 50
Epoch  200  loss  0.219205054414717 correct 50
Epoch  210  loss  0.21616939002144805 correct 50
Epoch  220  loss  0.2079560386964087 correct 50
Epoch  230  loss  0.1127355798957119 correct 50
Epoch  240  loss  0.24212786146210355 correct 50
Epoch  250  loss  0.018897605025617947 correct 50
Epoch  260  loss  0.14020023714668248 correct 50
Epoch  270  loss  0.09564912710582316 correct 50
Epoch  280  loss  0.03935236640668268 correct 50
Epoch  290  loss  0.29484622123434834 correct 50
Epoch  300  loss  0.15662750025756 correct 50
Epoch  310  loss  0.010451309606775455 correct 50
Epoch  320  loss  0.000614688771735292 correct 50
Epoch  330  loss  0.06471673605789798 correct 50
Epoch  340  loss  0.21566489097320035 correct 50
Epoch  350  loss  0.08764343668208285 correct 50
Epoch  360  loss  0.14923835852846995 correct 50
Epoch  370  loss  0.1046013379640636 correct 50
Epoch  380  loss  0.05277244015645201 correct 50
Epoch  390  loss  0.005926502060441047 correct 50
Epoch  400  loss  0.0911913382797357 correct 50
Epoch  410  loss  0.3155569958106067 correct 50
Epoch  420  loss  0.0039020848091212215 correct 50
Epoch  430  loss  0.09079889012837424 correct 50
Epoch  440  loss  0.021394666948184565 correct 50
Epoch  450  loss  0.13822232421269598 correct 50
Epoch  460  loss  0.03459683012521656 correct 50
Epoch  470  loss  0.08636568866990281 correct 50
Epoch  480  loss  0.0003039795122678232 correct 50
Epoch  490  loss  0.09492557993950121 correct 50
```

**GPU**
- `Hidden dimensions`: 100
- `Learning rate`: 0.05
- `Number of epochs`: 500

```bash
Epoch  0  loss  4.105738689712002 correct 28
Epoch  10  loss  2.319914863995853 correct 46
Epoch  20  loss  1.9965319931075822 correct 45
Epoch  30  loss  1.1333801227088982 correct 47
Epoch  40  loss  1.4033636823224922 correct 50
Epoch  50  loss  1.6687061428194092 correct 49
Epoch  60  loss  1.3248021501317058 correct 48
Epoch  70  loss  0.5172338717012002 correct 48
Epoch  80  loss  1.533717641053023 correct 48
Epoch  90  loss  2.0149492832413864 correct 48
Epoch  100  loss  1.923151734780004 correct 48
Epoch  110  loss  0.9066050937833137 correct 47
Epoch  120  loss  1.7521607442034726 correct 50
Epoch  130  loss  0.6528255146883242 correct 50
Epoch  140  loss  1.0278409462394622 correct 49
Epoch  150  loss  0.7301939946844127 correct 47
Epoch  160  loss  1.412902912089123 correct 47
Epoch  170  loss  0.41330497420223894 correct 50
Epoch  180  loss  1.867439732034504 correct 46
Epoch  190  loss  1.4650585986225226 correct 48
Epoch  200  loss  1.6155770757809509 correct 50
Epoch  210  loss  1.8048293195345742 correct 47
Epoch  220  loss  0.20712670812834966 correct 48
Epoch  230  loss  0.21698625277357778 correct 50
Epoch  240  loss  0.33461994961262215 correct 48
Epoch  250  loss  1.5014820853835265 correct 48
Epoch  260  loss  2.3028754917216165 correct 46
Epoch  270  loss  3.081872260502995 correct 46
Epoch  280  loss  0.6584209157344886 correct 50
Epoch  290  loss  0.7348878761623235 correct 50
Epoch  300  loss  0.9294512537436397 correct 48
Epoch  310  loss  1.264634722724447 correct 49
Epoch  320  loss  1.0470465543233411 correct 50
Epoch  330  loss  0.08579914342892472 correct 48
Epoch  340  loss  0.8852618212095407 correct 50
Epoch  350  loss  0.4267878919560012 correct 48
Epoch  360  loss  0.5916646848848208 correct 49
Epoch  370  loss  0.48873050553142894 correct 49
Epoch  380  loss  1.1138907430409142 correct 50
Epoch  390  loss  1.5843806420318862 correct 49
Epoch  400  loss  0.8566075862051498 correct 49
Epoch  410  loss  1.3578187324226927 correct 48
Epoch  420  loss  0.34565610833405636 correct 50
Epoch  430  loss  0.05697618773013966 correct 48
Epoch  440  loss  0.0019165475279466714 correct 50
Epoch  450  loss  0.3019312028553829 correct 50
Epoch  460  loss  0.4805790511646865 correct 48
Epoch  470  loss  0.8968567090593691 correct 50
Epoch  480  loss  0.5769199548861149 correct 50
Epoch  490  loss  0.28534002524587215 correct 50
```


### Classification of Split Model

**CPU:**
- `Hidden dimensions`: 100
- `Learning rate`: 0.05
- `Number of epochs`: 500

```bash
Epoch  0  loss  8.237081181632423 correct 32
Epoch  10  loss  6.014540179705802 correct 43
Epoch  20  loss  4.634835940267607 correct 45
Epoch  30  loss  4.54471544333551 correct 47
Epoch  40  loss  3.034966171802052 correct 45
Epoch  50  loss  2.8269804848086 correct 42
Epoch  60  loss  3.2450985236705194 correct 50
Epoch  70  loss  2.9840112160234957 correct 48
Epoch  80  loss  1.8809294333141937 correct 49
Epoch  90  loss  2.08537903685181 correct 49
Epoch  100  loss  2.452081877413707 correct 49
Epoch  110  loss  1.9007558762666157 correct 50
Epoch  120  loss  2.8863859321804037 correct 49
Epoch  130  loss  1.9412824581973565 correct 48
Epoch  140  loss  3.3841592922162635 correct 45
Epoch  150  loss  1.608005430242294 correct 49
Epoch  160  loss  0.8427616557352442 correct 49
Epoch  170  loss  0.6984388454866028 correct 48
Epoch  180  loss  1.0637565866402514 correct 49
Epoch  190  loss  1.9212890435345271 correct 49
Epoch  200  loss  1.307551106063289 correct 49
Epoch  210  loss  1.9431590261393747 correct 49
Epoch  220  loss  1.1594517620942129 correct 49
Epoch  230  loss  0.8848012649043949 correct 49
Epoch  240  loss  0.7548750216907237 correct 49
Epoch  250  loss  0.942016860331903 correct 48
Epoch  260  loss  1.319841838348911 correct 49
Epoch  270  loss  0.246643454068906 correct 50
Epoch  280  loss  1.0598915896948307 correct 49
Epoch  290  loss  1.0752095606358214 correct 49
Epoch  300  loss  0.7950174934510985 correct 50
Epoch  310  loss  0.9975616871164873 correct 50
Epoch  320  loss  0.6609165869781056 correct 46
Epoch  330  loss  0.7649080344530198 correct 49
Epoch  340  loss  2.1637555057799136 correct 46
Epoch  350  loss  0.4838163963197597 correct 49
Epoch  360  loss  0.11850457817121568 correct 49
Epoch  370  loss  0.5091044566336611 correct 49
Epoch  380  loss  1.6320075431312449 correct 49
Epoch  390  loss  0.636357472611704 correct 49
Epoch  400  loss  0.4500710560420677 correct 49
Epoch  410  loss  1.4681810201709498 correct 49
Epoch  420  loss  0.5325457020631111 correct 49
Epoch  430  loss  0.47047993071022604 correct 49
Epoch  440  loss  0.9428157280213557 correct 49
Epoch  450  loss  1.497214980899495 correct 50
Epoch  460  loss  0.8835199957321647 correct 47
Epoch  470  loss  0.6914230056666584 correct 49
Epoch  480  loss  0.35608384388947295 correct 49
Epoch  490  loss  0.7821677877877384 correct 49
```

**GPU:**
- `Hidden dimensions`: 100
- `Learning rate`: 0.05
- `Number of epochs`: 500
```bash
Epoch  0  loss  6.832420354734673 correct 35
Epoch  10  loss  5.721419830032776 correct 41
Epoch  20  loss  4.139444551250008 correct 41
Epoch  30  loss  4.213874770225451 correct 40
Epoch  40  loss  3.9758242192052946 correct 47
Epoch  50  loss  3.179241443440756 correct 47
Epoch  60  loss  2.471216088477429 correct 47
Epoch  70  loss  1.6417436160034513 correct 47
Epoch  80  loss  2.3452260313877566 correct 48
Epoch  90  loss  3.0724098147320973 correct 47
Epoch  100  loss  3.646140141900205 correct 48
Epoch  110  loss  5.458071949878313 correct 40
Epoch  120  loss  1.2022558391185227 correct 50
Epoch  130  loss  0.7832499144917744 correct 49
Epoch  140  loss  1.7960604877933353 correct 49
Epoch  150  loss  2.019454790259813 correct 48
Epoch  160  loss  0.8601967771786554 correct 50
Epoch  170  loss  0.6612350530896651 correct 50
Epoch  180  loss  2.4965353291006576 correct 48
Epoch  190  loss  0.3631280475696281 correct 50
Epoch  200  loss  0.8231121416938317 correct 49
Epoch  210  loss  0.5506204057398773 correct 50
Epoch  220  loss  1.0539960208117511 correct 50
Epoch  230  loss  0.9547498315843821 correct 50
Epoch  240  loss  0.33850286197761525 correct 50
Epoch  250  loss  0.6886352735634101 correct 50
Epoch  260  loss  0.47759007190084596 correct 50
Epoch  270  loss  0.5048886609286145 correct 46
Epoch  280  loss  1.1285184466155498 correct 50
Epoch  290  loss  0.2047792987200418 correct 50
Epoch  300  loss  0.5229968932077622 correct 50
Epoch  310  loss  0.37693511289262305 correct 50
Epoch  320  loss  0.8363475616864977 correct 49
Epoch  330  loss  0.3383997612810198 correct 50
Epoch  340  loss  0.4943022853448788 correct 49
Epoch  350  loss  0.16003734271265885 correct 50
Epoch  360  loss  0.47502321536832287 correct 50
Epoch  370  loss  0.5045188267696996 correct 49
Epoch  380  loss  0.03846714402138072 correct 49
Epoch  390  loss  0.30295776617117004 correct 50
Epoch  400  loss  0.6464713711664681 correct 50
Epoch  410  loss  0.1474140405706268 correct 49
Epoch  420  loss  0.2775331781382911 correct 48
Epoch  430  loss  0.16092975928993508 correct 50
Epoch  440  loss  0.6210495146270606 correct 50
Epoch  450  loss  0.1090167858982377 correct 49
Epoch  460  loss  0.3407230208868077 correct 48
Epoch  470  loss  0.35524675986272874 correct 50
Epoch  480  loss  0.5432295327253771 correct 50
Epoch  490  loss  0.45180773773663047 correct 50
```


### Classification of Xor Model
**CPU:**
- `Hidden dimensions`: 100
- `Learning rate`: 0.05
- `Number of epochs`: 500

```bash
Epoch  0  loss  6.7447520234648435 correct 32
Epoch  10  loss  6.048395307356559 correct 35
Epoch  20  loss  5.601465593203202 correct 42
Epoch  30  loss  4.786810522404697 correct 41
Epoch  40  loss  2.3912166811845643 correct 43
Epoch  50  loss  4.561950321738226 correct 43
Epoch  60  loss  3.107698381668941 correct 43
Epoch  70  loss  3.1664429644595775 correct 46
Epoch  80  loss  2.56340245997584 correct 44
Epoch  90  loss  1.7965195707506576 correct 43
Epoch  100  loss  3.6000384848637603 correct 45
Epoch  110  loss  2.582103049081835 correct 44
Epoch  120  loss  2.0777680998224284 correct 45
Epoch  130  loss  1.728603901124643 correct 46
Epoch  140  loss  1.7594673410233104 correct 48
Epoch  150  loss  2.258007876083318 correct 45
Epoch  160  loss  1.2335493135378697 correct 48
Epoch  170  loss  1.8790937564292287 correct 48
Epoch  180  loss  1.0973465365081978 correct 46
Epoch  190  loss  0.3761871207232688 correct 49
Epoch  200  loss  0.7409765339953596 correct 50
Epoch  210  loss  1.273575474905563 correct 47
Epoch  220  loss  1.459576816880749 correct 47
Epoch  230  loss  1.7701272459017185 correct 47
Epoch  240  loss  1.0989865770890912 correct 49
Epoch  250  loss  1.2359845716716495 correct 49
Epoch  260  loss  1.2348229277464873 correct 49
Epoch  270  loss  1.3759738127420265 correct 48
Epoch  280  loss  1.872368414939713 correct 49
Epoch  290  loss  2.0002648676564947 correct 49
Epoch  300  loss  2.248149093407065 correct 47
Epoch  310  loss  0.3632802489471282 correct 49
Epoch  320  loss  1.2383204218731922 correct 49
Epoch  330  loss  1.5738975085124423 correct 48
Epoch  340  loss  0.08840330587590536 correct 49
Epoch  350  loss  0.14686013739499884 correct 49
Epoch  360  loss  1.1709613606421259 correct 48
Epoch  370  loss  1.651678462857635 correct 49
Epoch  380  loss  0.07808751174916262 correct 49
Epoch  390  loss  1.2150935445623625 correct 50
Epoch  400  loss  1.2902613039918418 correct 49
Epoch  410  loss  1.2034105112669538 correct 50
Epoch  420  loss  0.4494703502199724 correct 49
Epoch  430  loss  0.422067060166575 correct 48
Epoch  440  loss  0.9654247981431509 correct 49
Epoch  450  loss  1.130445916505512 correct 49
Epoch  460  loss  1.0024161446168383 correct 48
Epoch  470  loss  1.0322339018503894 correct 49
Epoch  480  loss  0.12623919726895938 correct 49
Epoch  490  loss  0.21868738597858622 correct 50
```

**GPU:**
- `Hidden dimensions`: 100
- `Learning rate`: 0.05
- `Number of epochs`: 500
```bash
Epoch  0  loss  6.641016659911106 correct 32
Epoch  10  loss  5.302588267924449 correct 43
Epoch  20  loss  4.497547169205697 correct 46
Epoch  30  loss  3.5992638944826862 correct 47
Epoch  40  loss  3.125056959429007 correct 47
Epoch  50  loss  2.7445671159823157 correct 45
Epoch  60  loss  1.8739417045455997 correct 44
Epoch  70  loss  2.0518905953142252 correct 48
Epoch  80  loss  1.5152223092736736 correct 47
Epoch  90  loss  3.573875702252961 correct 47
Epoch  100  loss  2.0911094074872674 correct 47
Epoch  110  loss  2.715588253542184 correct 48
Epoch  120  loss  3.1305712600954663 correct 47
Epoch  130  loss  2.6555387137261244 correct 49
Epoch  140  loss  1.8881166311069388 correct 48
Epoch  150  loss  0.8726807957976102 correct 48
Epoch  160  loss  0.8874780659625571 correct 48
Epoch  170  loss  3.3857676788482025 correct 46
Epoch  180  loss  2.8327365960572752 correct 49
Epoch  190  loss  0.43089169979960307 correct 49
Epoch  200  loss  2.6898802331458853 correct 49
Epoch  210  loss  2.2954097165455436 correct 49
Epoch  220  loss  2.9756512605010856 correct 49
Epoch  230  loss  0.5580218054604333 correct 49
Epoch  240  loss  1.8638855676132784 correct 49
Epoch  250  loss  0.2175137962533066 correct 50
Epoch  260  loss  0.9120187983121234 correct 49
Epoch  270  loss  0.7762456929474704 correct 49
Epoch  280  loss  0.9045328649429079 correct 47
Epoch  290  loss  1.4246416398265005 correct 50
Epoch  300  loss  0.40298950739078643 correct 50
Epoch  310  loss  0.12086529233297896 correct 50
Epoch  320  loss  1.6429923188314102 correct 49
Epoch  330  loss  0.1640487505633586 correct 49
Epoch  340  loss  1.399130705510429 correct 49
Epoch  350  loss  0.7143384392605444 correct 50
Epoch  360  loss  0.9050994834800083 correct 50
Epoch  370  loss  0.19063860025904253 correct 50
Epoch  380  loss  0.7475012001364527 correct 50
Epoch  390  loss  0.6408314765218193 correct 50
Epoch  400  loss  0.5071157084193135 correct 50
Epoch  410  loss  1.028989456663848 correct 50
Epoch  420  loss  0.31938453101839864 correct 50
Epoch  430  loss  0.05632914744997156 correct 50
Epoch  440  loss  0.7097896264751271 correct 50
Epoch  450  loss  0.06042535292533012 correct 50
Epoch  460  loss  0.33875490797331137 correct 50
Epoch  470  loss  0.10970627618985244 correct 50
Epoch  480  loss  0.6842496591618389 correct 50
Epoch  490  loss  0.46613894738936534 correct 50
```
