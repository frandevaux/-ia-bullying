# ia-bullying

## Analisis de columnas

- Eliminar columna "record"
- Unificar las tres features de bullying en una sola: Hacer un "OR" entre las tres columnas. Tomar las que son null como si fueran "FALSE"
- Cambiar Yes y No, por 1 y 0.
- Custom_age cambiarlo de string a int
- Eliminar útlimas tres columnas: "were_obese", "were_underweight", "were_overweight" dado al gran número de nulls
