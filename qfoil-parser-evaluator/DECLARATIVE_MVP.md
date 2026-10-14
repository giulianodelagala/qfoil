# MVP: consultas declarativas sobre QFOIL

## Propósito

Este documento define el primer producto mínimo para una capa declarativa
sobre QFOIL. Su propósito no es crear una variante completa de SQL ni
reemplazar el lenguaje lógico de QFOIL. El objetivo es expresar de forma breve
la operación que hoy se realiza manualmente en
[`examples/example_mux.ipynb`](examples/example_mux.ipynb):

1. definir relaciones y condiciones sobre tuplas parciales en QFOIL;
2. buscar heurísticamente una asignación que las satisfaga;
3. usar una semilla para guiar la búsqueda;
4. usar el testigo obtenido como semilla de una iteración posterior;
5. devolver el testigo y métricas que permitan inspeccionarlo.

El MVP debe permitir experimentar con PSR, PCE/SCE y variantes definidas por
el usuario, sin introducir un optimizador SQL, un catálogo de tablas ni una
nueva semántica probabilística.

## Decisión de arquitectura

La capa declarativa vive dentro del paquete `qfoil`, como un *frontend* de
consultas pequeño. No es un proyecto ni un runtime paralelo. La distribución
actual ya ofrece la frontera correcta:

```python
from qfoil import QFOIL

qfoil = QFOIL(...)
```

La conexión propuesta es:

```text
consulta MVP
    -> parser de consultas
    -> Query AST
    -> compilador
    -> expresión QFOIL + parámetros de búsqueda
    -> QFOIL.grow_witness(...)
    -> QFOILEvaluator / SIGMA / backends probabilísticos
```

La capa nueva **no** debe importar directamente `QFOILEvaluator`,
`WitnessSearch`, PyJuice ni Z3. Sólo debe depender de la fachada pública
`QFOIL`. Esto conserva la separación actual entre:

- `qfoil/language/`: lenguaje lógico QFOIL, sus AST y parser;
- `qfoil/runtime/`: evaluación de AST y funciones;
- `qfoil/quantifiers/`: `SIGMA` y búsqueda de testigos;
- `qfoil/backends/`: PyJuice y Z3 opcional;
- `qfoil/queries/`: nuevo frontend declarativo del MVP.

La estructura inicial propuesta es:

```text
qfoil/
  queries/
    __init__.py
    ast.py          # Query, FindQuery, QueryContext, ReportItem
    parser.py       # gramática limitada del MVP
    compiler.py     # Query AST -> llamada a QFOIL
    executor.py     # ejecuta y construye QueryResult
    result.py       # QueryResult y metadatos de búsqueda
```

## Principio central: QFOIL conserva la semántica

`WHERE` contiene condiciones componibles: relaciones registradas, conectores
booleanos y expresiones QFOIL existentes. El compilador expande las relaciones
y una notación de propiedades limitada antes de invocar el parser QFOIL. La
consulta no inventa una segunda lógica para `FULL`, `<<<`, `EP`, `CEP`, `FREE`,
`SIZE`, `SIGMA`, ni para las funciones definidas por el usuario.

Por ejemplo, esta fórmula sigue siendo QFOIL:

```text
FULL(x) * (candidate <<< x) * (EP(candidate, c) >= delta)
```

Por ejemplo, `candidate.SIZE` es sólo una forma legible de escribir
`SIZE(candidate)`. Cuando la consulta declara `UNDER CLASS c`,
`candidate.EP` equivale a `EP(candidate, c)`. Del mismo modo,
`candidate IS SUFFICIENT EXPLANATION OF x` se expande al predicado QFOIL
registrado para esa relación. QFOIL conserva la semántica de la expresión
resultante.

Esta decisión permite reutilizar las definiciones del notebook, como `SE`,
`MINIMUM_PSR`, `SCE`, `DFS` o `PROB_DFS`. Los conectores `AND`, `OR` y `NOT`
del frontend sólo componen condiciones y se reducen a operadores QFOIL; no
constituyen una semántica lógica paralela.

## Sintaxis del MVP

El único comando de consulta es `FIND`. La forma canónica es:

```text
UNDER CLASS <clase>, CONFIDENCE >= <umbral>
FIND <variable> [, <variable> ...]
WHERE <condición>
[USING SEED <variable-de-semilla>,
       BUDGET <entero>, POOL <entero>, INNER_BUDGET <entero>]
[REPORT <propiedad o expresión QFOIL> [AS <nombre>], ...];
```

Por ejemplo:

```text
UNDER CLASS c, CONFIDENCE >= delta
FIND candidate
WHERE candidate IS SUFFICIENT EXPLANATION OF x
  AND candidate.SIZE <= size_max
USING BUDGET 1000, POOL 30;
```

Significado:

- `UNDER` establece el contexto semántico común. `CLASS c` selecciona la clase
  de la explicación y `CONFIDENCE >= delta` fija su requisito probabilístico.
  Ambos valores pueden ser literales o nombres tomados de los *bindings*.
- `FIND` liga una o más variables desconocidas y declara qué asignación debe
  devolver la consulta. No les asigna un tipo ni una noción privilegiada.
- `WHERE` es obligatorio y contiene toda la propiedad buscada. Puede combinar
  relaciones registradas, expresiones QFOIL y propiedades mediante `AND`,
  `OR`, `NOT` y paréntesis. Una relación puede aportar un dominio implícito; por
  ejemplo, `candidate IS SUFFICIENT EXPLANATION OF x` restringe `candidate` a
  las subtuplas de `x`.
- Un identificador es variable de búsqueda sólo si aparece en `FIND`. Los demás
  nombres de `WHERE`, como `x`, provienen de los *bindings* del programa
  anfitrión.
- `USING` contiene opciones operacionales y no cambia el significado lógico.
  `SEED incumbent` pasa el binding `incumbent` como `seed` a
  `QFOIL.grow_witness`; `BUDGET`, `POOL` e `INNER_BUDGET` corresponden a
  `budget`, `pool_size` e `inner_budget`.
- `REPORT` es opcional y agrega propiedades o proyecciones al reporte estándar.
  No reemplaza ni puede ocultar el testigo encontrado.

`UNDER` no contiene opciones de búsqueda: cambiar la clase o la confianza
cambia la propiedad consultada, mientras que cambiar el presupuesto sólo
cambia el esfuerzo de la heurística.

La forma `SEED incumbent` se refiere a la única variable de una consulta
monovariable. La futura ejecución multivariable deberá introducir semillas
identificadas por variable; esa sintaxis no forma parte del MVP inicial.

`FIND` significa encontrar una asignación testigo, no enumerar una relación
completa. No se admite todavía `SELECT`, `JOIN`, `GROUP BY`, `ORDER BY`,
subconsultas,
`INSERT`, `UPDATE`, `DELETE` ni varios generadores.

### Gramática indicativa

La gramática puede ser manual como el parser QFOIL actual o implementarse con
una librería, pero su alcance debe permanecer pequeño:

```ebnf
query             ::= context-clause? find-clause where-clause
                      using-clause? report-clause? ";"?
context-clause    ::= "UNDER" context-item ("," context-item)*
context-item      ::= "CLASS" value
                    | "CONFIDENCE" ">=" value
find-clause       ::= "FIND" identifier ("," identifier)*
where-clause      ::= "WHERE" condition
condition         ::= disjunction
disjunction       ::= conjunction ("OR" conjunction)*
conjunction       ::= negation ("AND" negation)*
negation          ::= "NOT" negation | primary-condition
primary-condition ::= "(" condition ")"
                    | relation
                    | query-expression
relation          ::= value "IS" registered-name "OF" value
using-clause      ::= "USING" search-option ("," search-option)*
search-option     ::= "SEED" identifier
                    | "BUDGET" integer
                    | "POOL" integer
                    | "INNER_BUDGET" integer
report-clause     ::= "REPORT" report-item ("," report-item)*
report-item       ::= query-expression ("AS" identifier)?
value             ::= identifier | literal
```

La precedencia, de mayor a menor, es `NOT`, `AND`, `OR`. Toda condición debe
evaluarse como `0` o `1`. El frontend compila cada relación por separado y
reduce `A AND B` a `A * B`, conserva `NOT(A)` y reduce `A OR B` a
`NOT(NOT(A) * NOT(B))`. Los paréntesis permiten hacer explícita cualquier
agrupación.

`query-expression` es una expresión QFOIL con azúcar sintáctico acotado para
propiedades. El frontend traduce, por ejemplo, `candidate.SIZE` a
`SIZE(candidate)` y `candidate.EP` a `EP(candidate, c)` bajo `CLASS c`; luego
delega el fragmento resultante a `QFOIL.parse`. No todas las funciones se
convierten automáticamente en propiedades: el MVP sólo admite propiedades cuya
traducción sea inequívoca.

`registered-name` termina antes de `OF` y debe corresponder a una relación
registrada. `SUFFICIENT EXPLANATION` y `CONTRASTIVE EXPLANATION` son las dos
relaciones iniciales. Cada relación define los roles de sus argumentos, el
contexto requerido, los dominios que impone y su expansión a QFOIL. Para una
explicación suficiente, `CONFIDENCE` se compila como `EP(subject, c)`; para una
explicación contrastiva se compila como `CEP(subject, c)`. Las nociones
definidas por el usuario se incorporan registrando nuevas relaciones con el
mismo contrato.

La sintaxis `FIND candidate AS ...` no forma parte del MVP. Aunque es breve
para una consulta aislada, convierte una noción en el tipo privilegiado de la
consulta y dificulta combinarla con otras. En el núcleo, todas las nociones son
condiciones de primer nivel dentro de `WHERE`.

### Reporte predeterminado

Una consulta sin `REPORT` produce un resultado completo y estable:

- `bindings` contiene una entrada por cada variable declarada en `FIND`;
- `properties` contiene las propiedades genéricas, como `size`, de cada
  variable encontrada;
- `relations` registra los átomos relacionales, su valor de verdad y sus
  métricas, evitando que un único nombre como `confidence` resulte ambiguo
  cuando una variable participa en varias nociones;
- también se incluyen el contexto semántico efectivo, las semillas utilizadas
  y las opciones de búsqueda efectivas.

Para una consulta monovariable, `result.witness` es un acceso conveniente al
único valor de `bindings`. El campo canónico es `bindings`, pues sigue siendo
válido para `FIND x, y`. La cláusula `REPORT` sólo agrega expresiones o alias al
reporte; no puede suprimir la asignación encontrada.

## Ejemplo: explicación suficiente

La semántica de la noción permanece en QFOIL. El frontend puede respaldar
`SUFFICIENT EXPLANATION` con una función QFOIL registrada como `SE` o componer
directamente la fórmula equivalente; no crea un evaluador alternativo.

La variable buscada se declara en `FIND`; la noción y las restricciones se
componen en `WHERE`:

```text
UNDER CLASS c, CONFIDENCE >= delta
FIND candidate
WHERE candidate IS SUFFICIENT EXPLANATION OF x
  AND candidate.SIZE <= size_max
USING BUDGET 1000, POOL 30;
```

Su ejecución equivale aproximadamente a:

```python
witness = qfoil.grow_witness(
    "FULL(x) * (candidate <<< x) * "
    "(EP(candidate, c) >= delta) * "
    "(1 <= SIZE(candidate)) * (SIZE(candidate) <= size_max)",
    var="candidate",
    context={"x": x, "c": c, "delta": delta, "size_max": size_max},
    budget=1000,
    pool_size=30,
)
```

El resultado no es sólo una tupla. Debe conservar información suficiente para
iterar y depurar:

```python
QueryResult(
    status="found",           # found | not_found
    bindings={"candidate": (None, 1, ...)},
    witness=(None, 1, ...),
    properties={"candidate": {"size": 2}},
    relations=[
        {
            "name": "SUFFICIENT EXPLANATION",
            "subject": "candidate",
            "object": "x",
            "satisfied": True,
            "confidence": 1.0,
        },
    ],
    seed_used=None,
    search={"budget": 1000, "pool_size": 30, "inner_budget": 8},
)
```

El reporte estándar incluye la asignación, las propiedades genéricas, las
relaciones verificadas y los metadatos de búsqueda. Por ello no se necesita
`RETURN candidate` ni un `REPORT` obligatorio. Si el usuario quiere una
proyección adicional o un alias, puede extender el reporte:

```text
UNDER CLASS c, CONFIDENCE >= delta
FIND candidate
WHERE candidate IS SUFFICIENT EXPLANATION OF x
  AND candidate.SIZE <= size_max
USING BUDGET 1000, POOL 30
REPORT candidate.EP AS posterior;
```

`REPORT` es aditivo: `posterior` se agrega a las propiedades predeterminadas y
el testigo sigue estando disponible en `result.witness`.

## Composición de nociones

Una misma variable puede participar en varias relaciones. Por ejemplo, una
explicación suficiente común para dos instancias se expresa sin crear una
construcción especial `MUTUAL` o `COMMON`:

```text
UNDER CLASS c, CONFIDENCE >= delta
FIND candidate
WHERE candidate IS SUFFICIENT EXPLANATION OF x
  AND candidate IS SUFFICIENT EXPLANATION OF y
  AND candidate.SIZE <= size_max
USING BUDGET 1000, POOL 30;
```

Su expansión QFOIL es, aproximadamente:

```text
FULL(x) * FULL(y) *
(candidate <<< x) * (candidate <<< y) *
(EP(candidate, c) >= delta) *
(1 <= SIZE(candidate)) * (SIZE(candidate) <= size_max)
```

Como `candidate` debe ser subtupla de ambas instancias, sólo puede conservar
posiciones observadas compatibles entre `x` e `y`. El compilador puede eliminar
conjuntos idénticos como la condición de confianza repetida, pero esta
simplificación no cambia la semántica.

Las nociones diferentes también se combinan de manera uniforme:

```text
UNDER CLASS c, CONFIDENCE >= delta
FIND candidate
WHERE candidate IS SUFFICIENT EXPLANATION OF x
  AND NOT(candidate IS CONTRASTIVE EXPLANATION OF y);
```

## Varias variables de búsqueda

La semántica del lenguaje admite una lista de variables en `FIND`:

```text
UNDER CLASS c, CONFIDENCE >= delta
FIND sufficient, contrastive
WHERE sufficient IS SUFFICIENT EXPLANATION OF x
  AND contrastive IS CONTRASTIVE EXPLANATION OF x
  AND sufficient.SIZE <= contrastive.SIZE;
```

Esta consulta busca una sola asignación conjunta, no dos resultados
independientes:

```text
∃ sufficient, contrastive:
    SE(x, sufficient, c, delta)
    AND SCE(x, contrastive, c, delta)
    AND SIZE(sufficient) <= SIZE(contrastive)
```

El resultado canónico conserva ambos valores:

```python
QueryResult(
    status="found",
    bindings={
        "sufficient": (...),
        "contrastive": (...),
    },
    properties={
        "sufficient": {"size": 2},
        "contrastive": {"size": 3},
    },
    relations=[...],
    search={...},
)
```

Declarar varias variables no autoriza una evaluación secuencial ingenua. Buscar
primero `sufficient` y luego `contrastive` con el primer valor fijado no
equivale, en general, a buscar el par: puede ser necesario retroceder y cambiar
ambos valores.

La fachada pública actual `QFOIL.grow_witness(..., var=...)` sólo busca una
variable. Por tanto, el parser y el AST del MVP representan una lista de
variables, pero el executor heurístico inicial exige exactamente una y devuelve
un error de capacidad claro para consultas de mayor aridad. La ejecución
multivariable requerirá una operación pública que busque y verifique una
asignación conjunta; no se simulará en la capa de consultas mediante llamadas
secuenciales a `grow_witness`.

El contrato futuro puede tomar una forma como:

```python
assignment = qfoil.find_assignment(
    expression,
    variables=["sufficient", "contrastive"],
    context=bindings,
    budget=1000,
)
```

y devolver un mapeo de nombres a tuplas. El nombre de la operación no es parte
del MVP, pero la capa declarativa no debe implementar por su cuenta una versión
privada de esta capacidad.

## Ejemplo: refinamiento iterativo con semillas

La iteración debe vivir en el programa anfitrión —notebook, CLI o Python— y
no en una construcción de control nueva del lenguaje. Esto mantiene el DSL
declarativo y pequeño.

Para buscar una SCE que libere menos atributos que una explicación conocida,
la propiedad es QFOIL:

```text
SCE(x, candidate, c, delta, release_max) *
NOT(FREE(x, incumbent) <= FREE(x, candidate))
```

La consulta usa el contexto para no repetir la clase y el umbral. La relación
contrastiva y las restricciones adicionales se combinan en el mismo `WHERE`:

```text
UNDER CLASS c, CONFIDENCE >= delta
FIND candidate
WHERE candidate IS CONTRASTIVE EXPLANATION OF x
  AND SIZE(FREE(x, candidate)) <= release_max
  AND NOT(FREE(x, incumbent) <= FREE(x, candidate))
USING SEED incumbent, BUDGET 1000
REPORT FREE(x, candidate) AS released;
```

Uso esperado desde Python:

```python
result = executor.execute(query, {
    "x": instance,
    "c": class_id,
    "delta": 0.9,
    "release_max": 8,
    "incumbent": initial_witness,
})

while result.found:
    incumbent = result.witness
    result = executor.execute(query, {
        "x": instance,
        "c": class_id,
        "delta": 0.9,
        "release_max": 8,
        "incumbent": incumbent,
    })
```

La semilla guía la heurística; la condición `NOT(FREE(...))` garantiza que un
resultado aceptado es estrictamente mejor que el incumbente. Si la búsqueda no
encuentra otro testigo, el último incumbente sigue siendo el mejor encontrado,
no una prueba global de optimalidad.

## Semántica y límites de corrección

El executor debe validar antes de iniciar la búsqueda:

- que las variables de `FIND` sean distintas y no estén ligadas previamente;
- que cada variable de búsqueda reciba un dominio mediante una relación
  positiva registrada o una restricción de dominio explícita. Una relación
  bajo `NOT` no aporta dominio; bajo `OR`, cada rama debe aportar dominios
  compatibles para todas sus variables libres;
- que todas las relaciones de `WHERE` estén registradas y reciban argumentos
  compatibles con sus roles;
- que todo identificador que no aparezca en `FIND` esté disponible en los
  *bindings* o sea parte del contexto declarado;
- que `UNDER` proporcione el contexto requerido por cada relación, incluida la
  clase para las propiedades probabilísticas;
- que el executor seleccionado soporte la cantidad de variables solicitada;
  el executor heurístico inicial sólo admite una;
- que `SEED` refiera a un *binding* de tipo tupla y longitud compatible;
- que `WHERE`, después de expandir relaciones, conectores y propiedades, pueda
  ser parseado por `QFOIL.parse`;
- que todos los elementos opcionales de `REPORT` puedan evaluarse con el
  conjunto de bindings encontrado y el contexto.

Después de la búsqueda, debe evaluar de nuevo la condición compilada completa
—dominios, relaciones, contexto `UNDER` y demás restricciones de `WHERE`— con
la asignación encontrada. Una asignación sólo se reporta como `found` si esa
comprobación es verdadera. Esto protege al usuario de resultados heurísticos
incompletos o de cambios futuros en la implementación de la búsqueda.

El MVP tiene dos resultados, no tres:

- `found`: se encontró y verificó una asignación testigo.
- `not_found`: el presupuesto heurístico se agotó sin una asignación
  verificada.

`not_found` no significa que no exista solución. Las pruebas universales y
las afirmaciones de optimalidad global quedan fuera del MVP, pues requieren
un protocolo explícito de prueba/contraejemplo y posiblemente Z3.
Una aridad no soportada es un error de capacidad, no un tercer estado de
resultado.

## Integración mínima de API

Una API deliberadamente pequeña basta para el primer hito:

```python
from qfoil.queries import QueryExecutor, QueryParser

query = QueryParser().parse(source)
executor = QueryExecutor(qfoil)
result = executor.execute(query, bindings)
```

No se añadirá otro evaluador ni se expondrán detalles de PyJuice o Z3 en la
consulta. El `QFOIL` recibido ya tiene configurado su backend probabilístico,
por ejemplo:

```python
qfoil = QFOIL(
    probabilistic_backend=PyJuiceProbabilityBackend(model),
    instance_len=8,
)
```

Por tanto, `EP`, `CEP`, `VAL` y `COND` conservan exactamente el contrato
actual de QFOIL. Una consulta que sólo use funciones de tuplas puede ejecutarse
sin backend probabilístico.

## Hitos de implementación

1. Definir `QueryContext`, `FindQuery`, `RelationCondition`, los nodos de
   composición booleana, `ReportItem` y `QueryResult`, con pruebas de AST.
   `FindQuery.variables` debe ser una lista aunque el primer executor sólo
   acepte una entrada.
2. Implementar el parser de `UNDER ... FIND ... WHERE ...`, incluidas las
   relaciones `IS ... OF`, los conectores `AND`/`OR`/`NOT`, los paréntesis y
   las cláusulas opcionales `USING` y `REPORT`.
3. Implementar `QueryExecutor` como una envoltura de `QFOIL.grow_witness`.
4. Implementar el registro de relaciones, su inferencia de dominio y su
   expansión a QFOIL, inicialmente para `SUFFICIENT EXPLANATION` y
   `CONTRASTIVE EXPLANATION`.
5. Implementar las traducciones de propiedades inequívocas, inicialmente
   `candidate.SIZE`, `candidate.EP` y `candidate.CEP`.
6. Añadir a `USING` las opciones `SEED`, `BUDGET`, `POOL` e `INNER_BUDGET`.
7. Construir `bindings`, las propiedades y relaciones del reporte
   predeterminado, y permitir que `REPORT` lo extienda.
8. Rechazar con un error de capacidad las consultas multivariables mientras la
   fachada pública no ofrezca búsqueda conjunta.
9. Verificar el predicado compilado y las proyecciones tras obtener el testigo.
10. Reproducir en pruebas el caso SE/PSR, la composición de una explicación
   común y una mejora iterativa SCE del
   notebook MUX, con un backend determinista pequeño cuando sea posible.
11. Añadir al notebook una celda que compare la llamada imperativa actual con
   la consulta declarativa equivalente.

El parser de consultas y su executor deben tener pruebas propias en
`tests/test_query_parser.py` y `tests/test_query_executor.py`. Las pruebas de
semántica de las fórmulas siguen perteneciendo a los tests actuales de QFOIL;
el MVP no debe duplicarlas.

## Criterio de salida del MVP

El MVP estará completo cuando el flujo siguiente funcione de punta a punta:

1. se registra una relación de explicación respaldada por una fórmula o función
   QFOIL como `SE` o `SCE`;
2. una consulta `UNDER ... FIND ... WHERE` combina una o más relaciones y
   restricciones sobre la misma variable;
3. el executor heurístico encuentra y verifica un testigo monovariable con
   `EP` o `CEP`, y rechaza explícitamente una aridad no soportada;
4. el resultado devuelve por defecto `bindings`, propiedades, relaciones y
   metadatos de búsqueda, sin requerir `REPORT`;
5. `REPORT`, cuando aparece, agrega propiedades sin reemplazar el reporte
   predeterminado;
6. una segunda consulta puede usar ese testigo como `SEED` e `incumbent`;
7. toda solución devuelta se verifica con el evaluador QFOIL.

Una vez estable este flujo, `COMPLETIONS`, búsqueda exacta/Z3, `CHECK`,
optimización declarativa, ejecución multivariable y nuevas relaciones pueden
añadirse sin cambiar la semántica ni la API básica de `FIND`.
