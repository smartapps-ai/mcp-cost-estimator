| Server     | Dataset   | Target        |   N Samples |   Train Mae |   Train Rmse |   Train R2 |   Train Mape |   Cv Mae |
|:-----------|:----------|:--------------|------------:|------------:|-------------:|-----------:|-------------:|---------:|
| sql_server | unitus    | input_tokens  |          27 |     23306.4 |      60686.7 |     -0.069 |        113.4 |  23040.3 |
| sql_server | unitus    | output_tokens |          27 |      1858.3 |       2659.9 |     -0.103 |         91.5 |   1971.5 |
| supabase   | unitus    | input_tokens  |          27 |      4049.6 |       5939.9 |     -0.01  |         29.7 |   4206.2 |
| supabase   | unitus    | output_tokens |          27 |      1643.7 |       2112.9 |     -0.108 |         84.7 |   1715.6 |
| tursio     | unitus    | input_tokens  |          27 |      2127.3 |       5204.3 |      0.187 |         22.8 |   2732.5 |
| tursio     | unitus    | output_tokens |          27 |       161.6 |        241.3 |      0.537 |         34.3 |    243.9 |
| snowflake  | umcu      | input_tokens  |          11 |       963.1 |       1354.4 |      0.984 |          3.2 |  12384.9 |
| snowflake  | umcu      | output_tokens |          11 |       394.8 |        457.5 |      0.971 |         16   |   1127.2 |
| supabase   | umcu      | input_tokens  |          11 |      2074.4 |       2514.6 |      0.851 |          3.6 |   4096.6 |
| supabase   | umcu      | output_tokens |          11 |        71.3 |        107   |      0.999 |          3.4 |   3583   |
| tursio     | umcu      | input_tokens  |          11 |     20661.2 |      40682.9 |     -0.043 |         35.2 |  25190.1 |
| tursio     | umcu      | output_tokens |          11 |       586.7 |       1145.9 |      0.662 |         36.9 |    847.9 |
| motherduck | tpch      | input_tokens  |          10 |       745.6 |       1010.7 |      0.943 |         12.8 |   3593.7 |
| motherduck | tpch      | output_tokens |          10 |       765.9 |        977.3 |     -0.061 |         58.1 |    827   |
| supabase   | tpch      | input_tokens  |          12 |      3656.7 |       4362.7 |      0.238 |         46   |   4232.2 |
| supabase   | tpch      | output_tokens |          12 |       564   |        712.7 |      0.64  |         38.1 |    813.9 |
| tursio     | tpch      | input_tokens  |          12 |      4222.6 |       7229.6 |     -0.072 |         55.7 |   4535.1 |
| tursio     | tpch      | output_tokens |          12 |       149.8 |        199.7 |      0.868 |         19.3 |    425.1 |