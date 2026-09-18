mod batch;
#[cfg(feature = "python")]
mod retry;
mod row_wise;

use std::sync::Arc;

pub use batch::{BatchPyFn, batch_udf};
use common_error::DaftResult;
use daft_core::prelude::*;
#[cfg(feature = "python")]
pub use retry::{retry_after_ms_from_error, retry_with_backoff};
pub use row_wise::{RowWisePyFn, row_wise_udf};
use serde::{Deserialize, Serialize};

use crate::{ExprRef, operator_metrics::MetricsCollector};

#[derive(derive_more::Display, Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[display("{_0}")]
pub enum PyScalarFn {
    RowWise(RowWisePyFn),
    Batch(BatchPyFn),
}

impl PyScalarFn {
    pub fn id(&self) -> Arc<str> {
        match self {
            Self::RowWise(func) => func.func_id.clone(),
            Self::Batch(func) => func.func_id.clone(),
        }
    }

    // pub fn name(&self) -> &str {
    //     match self {
    //         Self::RowWise(RowWisePyFn { function_name, .. })
    //         | Self::Batch(BatchPyFn { function_name, .. }) => function_name,
    //     }
    // }

    pub fn call(&self, args: &[Series], metrics: &mut dyn MetricsCollector) -> DaftResult<Series> {
        match self {
            Self::RowWise(func) => func.call(args, metrics),
            Self::Batch(func) => func.call(args, metrics),
        }
    }

    pub async fn call_async(
        &self,
        args: &[Series],
        metrics: &mut dyn MetricsCollector,
    ) -> DaftResult<Series> {
        match self {
            Self::RowWise(func) => func.call_async(args, metrics).await,
            Self::Batch(func) => func.call_async(args, metrics).await,
        }
    }

    pub fn args(&self) -> Vec<ExprRef> {
        match self {
            Self::RowWise(RowWisePyFn { args, .. }) | Self::Batch(BatchPyFn { args, .. }) => {
                args.clone()
            }
        }
    }

    pub fn to_field(&self, schema: &Schema) -> DaftResult<Field> {
        match self {
            Self::RowWise(RowWisePyFn {
                func_id,
                args,
                return_dtype,
                ..
            })
            | Self::Batch(BatchPyFn {
                func_id,
                args,
                return_dtype,
                ..
            }) => {
                let field_name = if let Some(first_child) = args.first() {
                    first_child.get_name(schema)?
                } else {
                    func_id.to_string()
                };

                Ok(Field::new(field_name, return_dtype.clone()))
            }
        }
    }

    pub fn with_new_children(&self, children: Vec<ExprRef>) -> Self {
        match self {
            Self::RowWise(row_wise_py_fn) => {
                Self::RowWise(row_wise_py_fn.with_new_children(children))
            }
            Self::Batch(batch_py_fn) => Self::Batch(batch_py_fn.with_new_children(children)),
        }
    }

    pub fn dtype(&self) -> DataType {
        match self {
            Self::RowWise(RowWisePyFn { return_dtype, .. })
            | Self::Batch(BatchPyFn { return_dtype, .. }) => return_dtype.clone(),
        }
    }

    pub fn is_async(&self) -> bool {
        match self {
            Self::RowWise(RowWisePyFn { is_async, .. }) => *is_async,
            Self::Batch(BatchPyFn { is_async, .. }) => *is_async,
        }
    }
}

#[cfg(feature = "python")]
pub fn collect_operator_metrics(
    operator_metrics: &common_metrics::python::PyOperatorMetrics,
    metrics: &mut dyn crate::operator_metrics::MetricsCollector,
) {
    for (name, counters) in operator_metrics.inner.snapshot() {
        for counter in counters {
            metrics.inc_counter(
                &name,
                counter.value,
                counter.description.as_deref(),
                Some(counter.attributes),
            );
        }
    }
}

/// Records the `udf.errors` / `udf.error_rows` counters for failures that `on_error`
/// turned into nulls instead of propagating.
///
/// `num_invocations` counts UDF calls that failed, `num_rows` counts the rows those
/// calls nulled out. The two differ for batch and async UDFs, where a single failure
/// nulls the whole batch.
#[cfg(feature = "python")]
pub(crate) fn record_suppressed_errors(
    metrics: &mut dyn crate::operator_metrics::MetricsCollector,
    function_name: &str,
    num_invocations: u64,
    num_rows: u64,
) {
    if num_invocations == 0 {
        return;
    }

    let attributes =
        std::collections::HashMap::from([("function".to_string(), function_name.to_string())]);

    metrics.inc_counter(
        common_metrics::UDF_ERRORS_KEY,
        num_invocations,
        Some("Number of Python UDF invocations suppressed by on_error"),
        Some(attributes.clone()),
    );
    metrics.inc_counter(
        common_metrics::UDF_ERROR_ROWS_KEY,
        num_rows,
        Some("Number of rows emitted as null by suppressed Python UDF invocations"),
        Some(attributes),
    );
}
