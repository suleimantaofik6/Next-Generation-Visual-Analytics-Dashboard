import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os

class DatasetAnalyzer:
    """
    A class for analyzing biomedical datasets and providing intelligent recommendations
    for visualization, dimensionality reduction, and machine learning.
    """
    
    def __init__(self, df=None):
        """Initialize with an optional DataFrame"""
        self.df = df
        self.analysis_results = {}
        
    def set_dataframe(self, df):
        """Set/update the DataFrame to analyze"""
        self.df = df
        # Reset analysis results when df changes
        self.analysis_results = {}
        
    def analyze_dataset(self):
        """Perform comprehensive analysis of the dataset"""
        try:
            if self.df is None:
                self.analysis_results = {"error": "No dataset loaded. Please upload a dataset first."}
                return self.analysis_results
            
            # Initialize results
            self.analysis_results = {}
            
            # Basic dataset properties - with error handling
            try:
                rows = len(self.df)
                cols = len(self.df.columns)
                missing_cells = self.df.isna().sum().sum()
                missing_percent = round((missing_cells / (rows * cols)) * 100, 2) if rows * cols > 0 else 0
                
                try:
                    memory_usage = round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)  # in MB
                except:
                    memory_usage = "Unknown"
                
                self.analysis_results["basic_info"] = {
                    "rows": rows,
                    "columns": cols,
                    "missing_cells": int(missing_cells),
                    "missing_percent": missing_percent,
                    "memory_usage": memory_usage
                }
            except Exception as e:
                self.analysis_results["basic_info"] = {
                    "rows": self.df.shape[0] if hasattr(self.df, 'shape') else 0,
                    "columns": self.df.shape[1] if hasattr(self.df, 'shape') else 0,
                    "error": f"Error computing basic stats: {str(e)}"
                }
            
            # Column types and statistics - with error handling
            try:
                # Try to get column types using pandas API
                numeric_cols = []
                categorical_cols = []
                boolean_cols = []
                datetime_cols = []
                
                try:
                    numeric_cols = list(self.df.select_dtypes(include=['number']).columns)
                except:
                    pass
                
                try:
                    categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
                except:
                    pass
                
                try:
                    boolean_cols = list(self.df.select_dtypes(include=['bool']).columns)
                except:
                    pass
                
                try:
                    datetime_cols = list(self.df.select_dtypes(include=['datetime']).columns)
                except:
                    pass
                
                # If standard approach fails, try alternate method to infer types
                if not numeric_cols and not categorical_cols and not boolean_cols and not datetime_cols:
                    for col in self.df.columns:
                        try:
                            if pd.api.types.is_numeric_dtype(self.df[col]):
                                numeric_cols.append(col)
                            elif pd.api.types.is_bool_dtype(self.df[col]):
                                boolean_cols.append(col)
                            elif pd.api.types.is_datetime64_dtype(self.df[col]):
                                datetime_cols.append(col)
                            else:
                                categorical_cols.append(col)
                        except:
                            # Default to categorical if we can't determine
                            categorical_cols.append(col)
                
                self.analysis_results["column_types"] = {
                    "numeric": len(numeric_cols),
                    "categorical": len(categorical_cols),
                    "boolean": len(boolean_cols),
                    "datetime": len(datetime_cols),
                    "all_columns": self.df.columns.tolist()
                }
            except Exception as e:
                self.analysis_results["column_types"] = {
                    "numeric": 0,
                    "categorical": 0,
                    "boolean": 0,
                    "datetime": 0,
                    "all_columns": self.df.columns.tolist() if hasattr(self.df, 'columns') else [],
                    "error": f"Error determining column types: {str(e)}"
                }
            
            # Detailed column analysis - each column in a try-except block
            self.analysis_results["columns"] = {}
            
            for col in self.df.columns:
                try:
                    col_data = {"name": col}
                    
                    # Determine column type
                    try:
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            col_data["type"] = "numeric"
                            non_null_values = self.df[col].dropna()
                            
                            if len(non_null_values) > 0:
                                try:
                                    col_data["min"] = float(non_null_values.min())
                                    col_data["max"] = float(non_null_values.max())
                                    col_data["mean"] = float(non_null_values.mean())
                                    col_data["median"] = float(non_null_values.median())
                                    col_data["std"] = float(non_null_values.std())
                                except Exception as stat_err:
                                    # If stats calculations fail, provide reasonable defaults
                                    col_data["stats_error"] = str(stat_err)
                                
                                try:
                                    # Check for binary/boolean numeric safely
                                    unique_values = non_null_values.unique()
                                    if len(unique_values) <= 2:
                                        col_data["likely_binary"] = True
                                except:
                                    pass
                            
                        elif pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                            col_data["type"] = "categorical"
                            non_null_values = self.df[col].dropna()
                            
                            if len(non_null_values) > 0:
                                try:
                                    unique_values = non_null_values.unique()
                                    col_data["unique_count"] = len(unique_values)
                                    
                                    # Only store unique values if there aren't too many
                                    if len(unique_values) <= 20:
                                        # Convert to list and ensure serializable
                                        safe_values = []
                                        for val in unique_values:
                                            try:
                                                # Ensure value is serializable
                                                safe_values.append(str(val))
                                            except:
                                                safe_values.append("Non-serializable value")
                                        col_data["unique_values"] = safe_values
                                        
                                    # Check if it might be a target variable
                                    if 2 <= len(unique_values) <= 15:
                                        col_data["potential_target"] = True
                                except Exception as e:
                                    col_data["unique_values_error"] = str(e)
                            
                        elif pd.api.types.is_datetime64_dtype(self.df[col]):
                            col_data["type"] = "datetime"
                            non_null_values = self.df[col].dropna()
                            
                            if len(non_null_values) > 0:
                                try:
                                    col_data["min"] = str(non_null_values.min())
                                    col_data["max"] = str(non_null_values.max())
                                    # Calculate time span safely
                                    if hasattr(col_data["max"] - col_data["min"], 'days'):
                                        col_data["time_span_days"] = (col_data["max"] - col_data["min"]).days
                                except Exception as dt_err:
                                    col_data["datetime_error"] = str(dt_err)
                        else:
                            # If type can't be determined, mark as unknown but still try to analyze
                            col_data["type"] = "unknown"
                    except Exception as type_err:
                        # If type determination fails completely
                        col_data["type"] = "unknown"
                        col_data["type_error"] = str(type_err)
                    
                    # Missing values analysis
                    try:
                        missing_count = self.df[col].isna().sum()
                        col_data["missing_count"] = int(missing_count)
                        col_data["missing_percent"] = round((missing_count / len(self.df)) * 100, 2) if len(self.df) > 0 else 0
                    except Exception as missing_err:
                        col_data["missing_error"] = str(missing_err)
                    
                    # Store column analysis
                    self.analysis_results["columns"][col] = col_data
                except Exception as col_err:
                    # Fallback for completely failed column analysis
                    self.analysis_results["columns"][col] = {
                        "name": col,
                        "type": "unknown",
                        "error": f"Error analyzing column: {str(col_err)}"
                    }
            
            # Analyze correlations for numeric data
            try:
                numeric_df = self.df.select_dtypes(include=['number'])
                if not numeric_df.empty and numeric_df.shape[1] > 1:
                    corr_matrix = numeric_df.corr().abs()
                    
                    # Find high correlations (excluding self-correlations)
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            try:
                                if corr_matrix.iloc[i, j] >= 0.7:  # High correlation threshold
                                    high_corr_pairs.append({
                                        "col1": str(corr_matrix.columns[i]),
                                        "col2": str(corr_matrix.columns[j]),
                                        "correlation": float(round(corr_matrix.iloc[i, j], 3))
                                    })
                            except:
                                # Skip this pair if there's an error
                                continue
                    
                    self.analysis_results["high_correlations"] = high_corr_pairs
            except Exception as e:
                self.analysis_results["high_correlations"] = []
                self.analysis_results["correlation_error"] = str(e)
            
            # Analyze dataset type and generate recommendations inside try blocks
            try:
                self._identify_dataset_type()
            except Exception as type_err:
                self.analysis_results["dataset_type_error"] = str(type_err)
                self.analysis_results["likely_dataset_type"] = "unknown"
            
            try:
                self._generate_recommendations()
            except Exception as rec_err:
                self.analysis_results["recommendations_error"] = str(rec_err)
                # Create empty recommendations to avoid KeyError
                self.analysis_results["recommendations"] = {
                    "visualizations": [],
                    "dimensionality_reduction": [],
                    "machine_learning": [],
                    "data_cleaning": []
                }
            
            return self.analysis_results
            
        except Exception as e:
            # Global error handler - ensure we always return something
            self.analysis_results = {
                "error": f"Error analyzing dataset: {str(e)}",
                "basic_info": {
                    "rows": self.df.shape[0] if hasattr(self.df, 'shape') else 0,
                    "columns": self.df.shape[1] if hasattr(self.df, 'shape') else 0,
                },
                "column_types": {
                    "numeric": 0,
                    "categorical": 0,
                    "boolean": 0,
                    "datetime": 0,
                    "all_columns": []
                },
                "columns": {},
                "recommendations": {
                    "visualizations": [],
                    "dimensionality_reduction": [],
                    "machine_learning": [],
                    "data_cleaning": []
                }
            }
            return self.analysis_results
    
    def _identify_dataset_type(self):
        """Identify the likely type of the dataset (classification, regression, etc.)"""
        dataset_types = []
        
        # Check for categorical columns that could be targets
        cat_cols = [col for col in self.analysis_results["columns"] 
                    if self.analysis_results["columns"][col].get("type") == "categorical" 
                    and self.analysis_results["columns"][col].get("potential_target")]
        
        if cat_cols:
            dataset_types.append("classification")
        
        # Check for numeric columns that might be continuous targets
        numeric_cols = [col for col in self.analysis_results["columns"] 
                      if self.analysis_results["columns"][col].get("type") == "numeric" 
                      and not self.analysis_results["columns"][col].get("likely_binary", False)]
        
        if numeric_cols:
            dataset_types.append("regression")
        
        # Check for datetime columns suggesting time series
        datetime_cols = [col for col in self.analysis_results["columns"] 
                       if self.analysis_results["columns"][col].get("type") == "datetime"]
        
        if datetime_cols:
            dataset_types.append("time_series")
        
        # Determine overall dataset type
        if "classification" in dataset_types and len(self.df.select_dtypes(include=['number']).columns) > 2:
            self.analysis_results["likely_dataset_type"] = "classification"
        elif "regression" in dataset_types and len(self.df.select_dtypes(include=['number']).columns) > 2:
            self.analysis_results["likely_dataset_type"] = "regression"
        elif "time_series" in dataset_types:
            self.analysis_results["likely_dataset_type"] = "time_series"
        else:
            self.analysis_results["likely_dataset_type"] = "exploratory" # General exploratory dataset
        
        self.analysis_results["possible_dataset_types"] = dataset_types
    
    def _generate_recommendations(self):
        """Generate intelligent recommendations based on the analysis"""
        recommendations = {}
        
        # Visualization recommendations
        recommendations["visualizations"] = self._recommend_visualizations()
        
        # Dimensionality reduction recommendations
        recommendations["dimensionality_reduction"] = self._recommend_dimensionality_reduction()
        
        # Machine learning recommendations
        recommendations["machine_learning"] = self._recommend_ml_models()
        
        # Data cleaning recommendations
        recommendations["data_cleaning"] = self._recommend_data_cleaning()
        
        self.analysis_results["recommendations"] = recommendations
    
    def _recommend_visualizations(self):
        """Recommend appropriate visualizations based on data types"""
        visualizations = []
        
        # Count column types
        num_numeric = self.analysis_results["column_types"]["numeric"]
        num_categorical = self.analysis_results["column_types"]["categorical"]
        
        # Get column names by type
        numeric_cols = [col for col in self.analysis_results["columns"] 
                       if self.analysis_results["columns"][col].get("type") == "numeric"]
        categorical_cols = [col for col in self.analysis_results["columns"] 
                           if self.analysis_results["columns"][col].get("type") == "categorical"]
        
        # For datasets with a mix of numeric and categorical
        if num_numeric > 0 and num_categorical > 0:
            visualizations.append({
                "type": "box_plot",
                "description": "Box plots to show distribution of numeric variables across categories",
                "suitable_columns": {
                    "x": categorical_cols[:min(3, len(categorical_cols))],
                    "y": numeric_cols[:min(5, len(numeric_cols))]
                }
            })
            
            visualizations.append({
                "type": "bar_chart",
                "description": "Bar charts to compare numeric values across categories",
                "suitable_columns": {
                    "x": categorical_cols[:min(3, len(categorical_cols))],
                    "y": numeric_cols[:min(5, len(numeric_cols))]
                }
            })
        
        # For datasets with multiple numeric columns
        if num_numeric >= 2:
            visualizations.append({
                "type": "scatter_plot",
                "description": "Scatter plots to visualize relationships between numeric variables",
                "suitable_columns": {
                    "x": numeric_cols[:min(len(numeric_cols), 5)],
                    "y": numeric_cols[:min(len(numeric_cols), 5)]
                }
            })
            
            visualizations.append({
                "type": "correlation_heatmap",
                "description": "Correlation heatmap to show relationships between all numeric variables",
                "suitable_columns": numeric_cols[:min(len(numeric_cols), 15)]
            })
            
            if num_numeric >= 3:
                visualizations.append({
                    "type": "3d_scatter",
                    "description": "3D scatter plot for exploring relationships among three numeric variables",
                    "suitable_columns": {
                        "x": numeric_cols[0] if len(numeric_cols) > 0 else None,
                        "y": numeric_cols[1] if len(numeric_cols) > 1 else None,
                        "z": numeric_cols[2] if len(numeric_cols) > 2 else None
                    }
                })
        
        # For datasets with categorical columns
        if num_categorical >= 1:
            visualizations.append({
                "type": "count_plot",
                "description": "Count plots to show frequency of categories",
                "suitable_columns": categorical_cols[:min(5, len(categorical_cols))]
            })
            
            if num_categorical >= 2:
                visualizations.append({
                    "type": "heatmap",
                    "description": "Heatmap to show relationship between two categorical variables",
                    "suitable_columns": {
                        "x": categorical_cols[0] if len(categorical_cols) > 0 else None,
                        "y": categorical_cols[1] if len(categorical_cols) > 1 else None
                    }
                })
        
        # For datasets with a target variable and numeric features
        potential_targets = [col for col in self.analysis_results["columns"] 
                            if self.analysis_results["columns"][col].get("potential_target", False)]
        
        if potential_targets and num_numeric > 0:
            visualizations.append({
                "type": "feature_importance",
                "description": "Feature importance plots to see which variables may predict the target",
                "target": potential_targets[0] if potential_targets else None,
                "features": numeric_cols[:min(len(numeric_cols), 10)]
            })
        
        return visualizations
    
    def _recommend_dimensionality_reduction(self):
        """Recommend dimensionality reduction techniques and parameters"""
        recommendations = []
        num_samples = self.analysis_results["basic_info"]["rows"]
        num_numeric = self.analysis_results["column_types"]["numeric"]
        
        # PCA recommendations
        if num_numeric >= 3:
            pca_rec = {
                "method": "PCA",
                "suitability": "high" if num_numeric > 10 else "medium",
                "description": "PCA is good for linear dimensionality reduction and works well with continuous, correlated features"
            }
            
            # Recommend components based on dataset size
            if num_numeric > 10:
                pca_rec["recommended_components"] = min(10, num_numeric // 2)
            else:
                pca_rec["recommended_components"] = 2
                
            # Add explanation
            pca_rec["explanation"] = (
                f"Recommend using {pca_rec['recommended_components']} components to capture most variance "
                f"while significantly reducing dimensions from original {num_numeric} features."
            )
            
            recommendations.append(pca_rec)
        
        # t-SNE recommendations
        tsne_rec = {
            "method": "t-SNE",
            "suitability": "high" if 50 <= num_samples <= 10000 else "medium",
            "description": "t-SNE is excellent for visualization and preserving local relationships in the data"
        }
        
        # Perplexity recommendations based on sample size
        if num_samples < 100:
            tsne_rec["perplexity"] = 5
            tsne_rec["n_iter"] = 1000
        elif num_samples < 1000:
            tsne_rec["perplexity"] = 30
            tsne_rec["n_iter"] = 1000
        else:
            tsne_rec["perplexity"] = 50
            tsne_rec["n_iter"] = 2000
            
        tsne_rec["explanation"] = (
            f"For your dataset with {num_samples} samples, recommended perplexity of {tsne_rec['perplexity']} "
            f"balances local and global structure, with {tsne_rec['n_iter']} iterations for convergence."
        )
        
        recommendations.append(tsne_rec)
        
        # UMAP recommendations
        umap_rec = {
            "method": "UMAP",
            "suitability": "high",
            "description": "UMAP is faster than t-SNE and often preserves more global structure while maintaining local relationships"
        }
        
        # Parameters based on dataset size
        if num_samples < 100:
            umap_rec["n_neighbors"] = 5
            umap_rec["min_dist"] = 0.1
        elif num_samples < 1000:
            umap_rec["n_neighbors"] = 15
            umap_rec["min_dist"] = 0.1
        else:
            umap_rec["n_neighbors"] = 30
            umap_rec["min_dist"] = 0.3
            
        umap_rec["explanation"] = (
            f"For your dataset with {num_samples} samples, recommended n_neighbors of {umap_rec['n_neighbors']} "
            f"and min_dist of {umap_rec['min_dist']} to balance local and global structure preservation."
        )
        
        recommendations.append(umap_rec)
        
        # LDA recommendations (only if classification task)
        potential_targets = [col for col in self.analysis_results["columns"] 
                             if self.analysis_results["columns"][col].get("potential_target", False)]
        
        if potential_targets and "classification" in self.analysis_results.get("possible_dataset_types", []):
            target_col = potential_targets[0]
            unique_classes = self.analysis_results["columns"][target_col].get("unique_count", 0)
            
            if 2 <= unique_classes <= 15:  # LDA works for classification with distinct classes
                lda_rec = {
                    "method": "LDA (supervised)",
                    "suitability": "high" if unique_classes > 2 else "medium",
                    "description": "LDA is excellent for supervised dimensionality reduction, finding components that maximize class separation"
                }
                
                # Max LDA components is min(n_features, n_classes-1)
                lda_rec["recommended_components"] = min(num_numeric, unique_classes - 1)
                lda_rec["target_column"] = target_col
                
                lda_rec["explanation"] = (
                    f"Since your data has {unique_classes} classes, LDA can reduce to max {lda_rec['recommended_components']} "
                    f"components while maximizing separation between classes in '{target_col}'."
                )
                
                recommendations.append(lda_rec)
        
        return recommendations
    
    def _recommend_ml_models(self):
        """Recommend machine learning models based on dataset characteristics"""
        recommendations = []
        
        # Determine if it's a classification task
        potential_targets = [col for col in self.analysis_results["columns"] 
                            if self.analysis_results["columns"][col].get("potential_target", False)]
        
        is_classification = ("classification" in self.analysis_results.get("possible_dataset_types", []) 
                             and potential_targets)
        
        is_regression = "regression" in self.analysis_results.get("possible_dataset_types", [])
        
        num_samples = self.analysis_results["basic_info"]["rows"]
        num_features = self.analysis_results["column_types"]["numeric"]
        
        # Default models list if we can't determine the task
        if not (is_classification or is_regression):
            return [{
                "message": "Unable to determine specific task type. Please verify if dataset contains target variables."
            }]
        
        # For classification tasks
        if is_classification:
            target_col = potential_targets[0]
            target_classes = self.analysis_results["columns"][target_col].get("unique_count", 0)
            
            # Binary classification
            if target_classes == 2:
                recommendations.append({
                    "model": "Logistic Regression",
                    "suitability": "high" if num_features < 50 and num_samples > 50 else "medium",
                    "hyperparameters": {
                        "C": 1.0,
                        "class_weight": "balanced" if self._check_class_imbalance(target_col) else None
                    },
                    "explanation": "Simple, interpretable model for binary classification; works well when relationship is linear"
                })
                
                recommendations.append({
                    "model": "Random Forest",
                    "suitability": "high",
                    "hyperparameters": {
                        "n_estimators": min(100, max(10, num_samples // 10)),
                        "max_depth": None,
                        "min_samples_split": 2,
                        "class_weight": "balanced" if self._check_class_imbalance(target_col) else None
                    },
                    "explanation": "Robust to overfitting, handles non-linear relationships well, can capture complex interactions"
                })
                
                recommendations.append({
                    "model": "Support Vector Machine (SVM)",
                    "suitability": "high" if 50 < num_samples < 10000 else "medium",
                    "hyperparameters": {
                        "C": 1.0,
                        "kernel": "rbf",
                        "class_weight": "balanced" if self._check_class_imbalance(target_col) else None
                    },
                    "explanation": "Effective for complex decision boundaries; works well in high dimensional spaces"
                })
                
                recommendations.append({
                    "model": "XGBoost",
                    "suitability": "high" if num_samples > 100 else "medium",
                    "hyperparameters": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 3
                    },
                    "explanation": "State-of-the-art gradient boosting model, highly accurate, efficient implementation"
                })
                
            # Multi-class classification  
            elif target_classes > 2:
                recommendations.append({
                    "model": "Random Forest",
                    "suitability": "high",
                    "hyperparameters": {
                        "n_estimators": min(100, max(10, num_samples // 10)),
                        "max_depth": None,
                        "min_samples_split": 2,
                        "class_weight": "balanced" if self._check_class_imbalance(target_col) else None
                    },
                    "explanation": "Handles multi-class problems naturally, robust to overfitting, good for complex relationships"
                })
                
                recommendations.append({
                    "model": "XGBoost",
                    "suitability": "high" if num_samples > 100 else "medium",
                    "hyperparameters": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 3
                    },
                    "explanation": "Excellent performance on multi-class problems, handles imbalanced classes well"
                })
                
                recommendations.append({
                    "model": "Support Vector Machine (SVM)",
                    "suitability": "medium" if 50 < num_samples < 5000 else "low",
                    "hyperparameters": {
                        "C": 1.0,
                        "kernel": "rbf" 
                    },
                    "explanation": "Uses one-vs-rest for multi-class problems, effective but can be computationally expensive"
                })
                
                recommendations.append({
                    "model": "K-Nearest Neighbors (KNN)",
                    "suitability": "medium" if num_samples < 10000 and num_features < 20 else "low",
                    "hyperparameters": {
                        "n_neighbors": min(int(np.sqrt(num_samples)), 20),
                        "weights": "distance"
                    },
                    "explanation": "Simple approach that can work well for multi-class, but sensitive to feature scaling and curse of dimensionality"
                })
        
        # For regression tasks
        elif is_regression:
            recommendations.append({
                "model": "Random Forest Regressor",
                "suitability": "high",
                "hyperparameters": {
                    "n_estimators": min(100, max(10, num_samples // 10)),
                    "max_depth": None
                },
                "explanation": "Handles non-linear relationships, robust to outliers, good with high-dimensional data"
            })
            
            recommendations.append({
                "model": "XGBoost Regressor",
                "suitability": "high" if num_samples > 100 else "medium",
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3
                },
                "explanation": "State-of-the-art regression performance, handles missing values well"
            })
            
            recommendations.append({
                "model": "Linear Regression",
                "suitability": "high" if self._check_linear_relationship() else "medium",
                "explanation": "Simple, fast, interpretable model for linear relationships between features and target"
            })
        
        return recommendations
    
    def _recommend_data_cleaning(self):
        """Recommend data cleaning steps based on analysis"""
        recommendations = []
        
        # Safety check - if basic_info doesn't exist for some reason, return empty recommendations
        if "basic_info" not in self.analysis_results:
            return recommendations
            
        missing_percent = self.analysis_results["basic_info"].get("missing_percent", 0)
        
        # Check for missing values
        if missing_percent > 0:
            recommendations.append({
                "issue": "missing_values",
                "description": f"Dataset contains {missing_percent}% missing values",
                "suggestion": "Handle missing values before analysis"
            })
            
            # Add column-specific recommendations
            cols_with_missing = [col for col in self.analysis_results["columns"] 
                               if self.analysis_results["columns"][col].get("missing_count", 0) > 0]
            
            for col in cols_with_missing:
                # Get column data with safety checks
                col_data = self.analysis_results["columns"].get(col, {})
                col_missing_pct = col_data.get("missing_percent", 0)
                col_type = col_data.get("type", "unknown")
                
                if col_missing_pct > 30:
                    recommendations.append({
                        "issue": f"high_missing_values_in_{col}",
                        "description": f"Column '{col}' has {col_missing_pct}% missing values",
                        "suggestion": f"Consider dropping this column as it has too many missing values"
                    })
                elif col_type == "numeric":
                    recommendations.append({
                        "issue": f"missing_values_in_{col}",
                        "description": f"Numeric column '{col}' has {col_missing_pct}% missing values",
                        "suggestion": f"Impute missing values with mean, median, or use advanced techniques like KNN imputation"
                    })
                elif col_type == "categorical":
                    recommendations.append({
                        "issue": f"missing_values_in_{col}",
                        "description": f"Categorical column '{col}' has {col_missing_pct}% missing values",
                        "suggestion": f"Impute missing values with mode or create a 'Missing' category"
                    })
        
        # Check for potential outliers in numeric columns
        if "columns" in self.analysis_results:
            for col, col_data in self.analysis_results["columns"].items():
                # Use safe dict access with get() to avoid KeyError
                if col_data.get("type") == "numeric":
                    std = col_data.get("std")
                    mean = col_data.get("mean")
                    
                    if std is not None and mean is not None and mean != 0:
                        if std > 0 and abs(std / mean) > 1.5:  # Rough heuristic for potential outliers
                            recommendations.append({
                                "issue": f"potential_outliers_in_{col}",
                                "description": f"Column '{col}' may contain outliers (high variance relative to mean)",
                                "suggestion": f"Check '{col}' for outliers using box plots or z-scores"
                            })
        
        # Check for highly correlated features
        if "high_correlations" in self.analysis_results and self.analysis_results["high_correlations"]:
            high_corr_count = len(self.analysis_results["high_correlations"])
            recommendations.append({
                "issue": "correlated_features",
                "description": f"Found {high_corr_count} pairs of highly correlated features (r ≥ 0.7)",
                "suggestion": "Consider removing one feature from each highly correlated pair to reduce multicollinearity"
            })
        
        return recommendations
    
    def _check_class_imbalance(self, target_col):
        """Check if target column has imbalanced classes"""
        if self.df is None or target_col not in self.df.columns:
            return False
        
        value_counts = self.df[target_col].value_counts()
        if len(value_counts) < 2:
            return False
            
        # Calculate ratio of most common to least common class
        imbalance_ratio = value_counts.max() / value_counts.min()
        
        # Consider imbalanced if ratio > 3
        return imbalance_ratio > 3
    
    def _check_linear_relationship(self):
        """Rough check for linear relationships in the data"""
        # This is a very simplified check - just look for high correlations
        if "high_correlations" in self.analysis_results:
            return len(self.analysis_results["high_correlations"]) > 0
        return False
    
    def generate_summary(self):
        """Generate a textual summary of the dataset analysis"""
        if not self.analysis_results:
            try:
                self.analyze_dataset()
            except Exception as e:
                return f"Error analyzing dataset: {str(e)}"
            
        try:
            # Get basic info with safety checks
            if "basic_info" not in self.analysis_results:
                return "Dataset analysis not complete. Please try analyzing the dataset again."
                
            basic = self.analysis_results["basic_info"]
            rows = basic.get("rows", "Unknown")
            columns = basic.get("columns", "Unknown")
            missing_percent = basic.get("missing_percent", 0)
            
            # Get column types with safety checks
            if "column_types" not in self.analysis_results:
                return f"Dataset has {rows} rows and {columns} columns.\nFull analysis not available."
                
            col_types = self.analysis_results["column_types"]
            num_numeric = col_types.get("numeric", 0)
            num_categorical = col_types.get("categorical", 0)
            
            summary = [
                f"Dataset has {rows} rows and {columns} columns.",
                f"Contains {num_numeric} numeric, {num_categorical} categorical columns.",
                f"Missing data: {missing_percent}% of cells are missing."
            ]
            
            if "likely_dataset_type" in self.analysis_results:
                summary.append(f"This appears to be a {self.analysis_results['likely_dataset_type']} dataset.")
            
            if "high_correlations" in self.analysis_results and self.analysis_results["high_correlations"]:
                correlation_count = len(self.analysis_results["high_correlations"])
                summary.append(f"Found {correlation_count} pairs of highly correlated features.")
            
            if "recommendations" in self.analysis_results:
                recom = self.analysis_results["recommendations"]
                
                if "dimensionality_reduction" in recom:
                    top_dim_method = None
                    top_suitability = "low"
                    for method in recom["dimensionality_reduction"]:
                        if method.get("suitability", "low") == "high":
                            top_dim_method = method.get("method", "Unknown")
                            top_suitability = "high"
                            break
                        elif method.get("suitability", "low") == "medium" and top_suitability != "high":
                            top_dim_method = method.get("method", "Unknown")
                            top_suitability = "medium"
                    
                    if top_dim_method:
                        summary.append(f"Recommended dimensionality reduction: {top_dim_method}.")
                
                if "machine_learning" in recom:
                    top_ml_model = None
                    top_suitability = "low"
                    for model in recom["machine_learning"]:
                        if isinstance(model, dict) and "model" in model:
                            if model.get("suitability", "low") == "high":
                                top_ml_model = model.get("model", "Unknown")
                                top_suitability = "high"
                                break
                            elif model.get("suitability", "low") == "medium" and top_suitability != "high":
                                top_ml_model = model.get("model", "Unknown")
                                top_suitability = "medium"
                    
                    if top_ml_model:
                        summary.append(f"Recommended machine learning model: {top_ml_model}.")
            
            return "\n".join(summary)
            
        except Exception as e:
            # Provide a basic summary with error information if anything fails
            rows = self.df.shape[0] if self.df is not None else "Unknown"
            cols = self.df.shape[1] if self.df is not None else "Unknown"
            return f"Dataset has {rows} rows and {cols} columns.\nError generating detailed summary: {str(e)}"
    
    def get_dataset_json_for_llm(self):
        """Return a simplified JSON representation of the dataset for LLM context"""
        try:
            if not self.analysis_results:
                try:
                    self.analyze_dataset()
                except Exception as e:
                    # If analysis fails, return error info
                    return json.dumps({
                        "error": f"Could not analyze dataset: {str(e)}",
                        "basic_info": {
                            "rows": str(self.df.shape[0]) if self.df is not None else "Unknown",
                            "columns": str(self.df.shape[1]) if self.df is not None else "Unknown"
                        }
                    })
                
            # Create a simplified version to avoid token limits
            simple_analysis = {}
            
            # Add basic info if available
            if "basic_info" in self.analysis_results:
                simple_analysis["basic_info"] = self._convert_to_json_serializable(
                    self.analysis_results["basic_info"]
                )
            else:
                # Use dataframe shape as fallback
                simple_analysis["basic_info"] = {
                    "rows": self.df.shape[0] if self.df is not None else "Unknown",
                    "columns": self.df.shape[1] if self.df is not None else "Unknown"
                }
            
            # Add column types if available
            if "column_types" in self.analysis_results:
                simple_analysis["column_types"] = self._convert_to_json_serializable(
                    self.analysis_results["column_types"]
                )
            
            # Add dataset type if available
            simple_analysis["likely_dataset_type"] = self.analysis_results.get("likely_dataset_type", "unknown")
            
            # Include top 10 columns information if available
            simple_analysis["columns"] = {}
            if "columns" in self.analysis_results:
                columns_to_include = list(self.analysis_results["columns"].keys())[:10]
                for col in columns_to_include:
                    simple_analysis["columns"][col] = self._convert_to_json_serializable(
                        self.analysis_results["columns"].get(col, {})
                    )
            
            # Include only top recommendations if available
            if "recommendations" in self.analysis_results:
                simple_analysis["recommendations"] = {}
                recom = self.analysis_results["recommendations"]
                
                # Add dimensionality reduction recommendations
                if "dimensionality_reduction" in recom:
                    simple_analysis["recommendations"]["dimensionality_reduction"] = []
                    for item in recom["dimensionality_reduction"][:2]:
                        if item.get("suitability", "") in ["high", "medium"]:
                            simple_analysis["recommendations"]["dimensionality_reduction"].append(
                                self._convert_to_json_serializable(item)
                            )
                
                # Add machine learning recommendations
                if "machine_learning" in recom:
                    simple_analysis["recommendations"]["machine_learning"] = []
                    for item in recom["machine_learning"][:2]:
                        if isinstance(item, dict) and item.get("suitability", "") in ["high", "medium"]:
                            simple_analysis["recommendations"]["machine_learning"].append(
                                self._convert_to_json_serializable(item)
                            )
            
            # Serialize to JSON with fallback for any JSON errors
            try:
                return json.dumps(simple_analysis, default=str)
            except Exception as e:
                # If serialization fails, return error info
                return json.dumps({
                    "error": f"Could not serialize dataset analysis: {str(e)}",
                    "basic_info": {
                        "rows": str(simple_analysis.get("basic_info", {}).get("rows", "Unknown")),
                        "columns": str(simple_analysis.get("basic_info", {}).get("columns", "Unknown"))
                    }
                }, default=str)
                
        except Exception as e:
            # Catch-all for any other errors
            return json.dumps({
                "error": f"Error in dataset analysis: {str(e)}",
                "message": "The AI assistant can still help with general questions about data science and machine learning."
            }, default=str)
    
    def _convert_to_json_serializable(self, obj):
        """Convert objects to JSON serializable types recursively"""
        if obj is None:
            return None
        
        # Handle basic data types
        if isinstance(obj, (str, bool)):
            return obj
            
        if isinstance(obj, (int, float)) and not hasattr(obj, 'dtype'):
            # Regular Python int/float, not numpy types
            return obj
        
        # Handle numpy numeric types
        if hasattr(obj, 'dtype') or hasattr(obj, 'item'):
            try:
                return obj.item()  # Convert numpy types to native Python types
            except:
                # If item() method fails, try direct conversion
                return float(obj) if 'float' in str(type(obj)).lower() else int(obj)
        
        # Handle lists, tuples, and numpy arrays
        if isinstance(obj, (list, tuple)) or (hasattr(obj, '__iter__') and not isinstance(obj, (str, dict))):
            return [self._convert_to_json_serializable(item) for item in obj]
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {
                str(key): self._convert_to_json_serializable(value) 
                for key, value in obj.items()
            }
        
        # Handle pandas DataFrame or Series
        if hasattr(obj, 'to_dict'):
            try:
                return self._convert_to_json_serializable(obj.to_dict())
            except:
                pass
                
        # Try converting everything else to string
        try:
            return str(obj)
        except:
            return "Value not serializable"
    
    def get_dimensionality_reduction_recommendations(self):
        """Get specific dimensionality reduction recommendations"""
        try:
            if not self.analysis_results or "recommendations" not in self.analysis_results:
                try:
                    self.analyze_dataset()
                except Exception as e:
                    return [{"error": f"Could not analyze dataset: {str(e)}"}]
                
            if "recommendations" in self.analysis_results and "dimensionality_reduction" in self.analysis_results["recommendations"]:
                # Convert to JSON serializable format
                recommendations = self.analysis_results["recommendations"]["dimensionality_reduction"]
                return self._convert_to_json_serializable(recommendations)
            return []
        except Exception as e:
            return [{"error": f"Error getting dimensionality reduction recommendations: {str(e)}"}]
    
    def get_ml_model_recommendations(self):
        """Get specific machine learning model recommendations"""
        try:
            if not self.analysis_results or "recommendations" not in self.analysis_results:
                try:
                    self.analyze_dataset()
                except Exception as e:
                    return [{"error": f"Could not analyze dataset: {str(e)}"}]
                
            if "recommendations" in self.analysis_results and "machine_learning" in self.analysis_results["recommendations"]:
                # Convert to JSON serializable format
                recommendations = self.analysis_results["recommendations"]["machine_learning"]
                return self._convert_to_json_serializable(recommendations)
            return []
        except Exception as e:
            return [{"error": f"Error getting machine learning model recommendations: {str(e)}"}]
    
    def get_visualization_recommendations(self):
        """Get specific visualization recommendations"""
        try:
            if not self.analysis_results or "recommendations" not in self.analysis_results:
                try:
                    self.analyze_dataset()
                except Exception as e:
                    return [{"error": f"Could not analyze dataset: {str(e)}"}]
                
            if "recommendations" in self.analysis_results and "visualizations" in self.analysis_results["recommendations"]:
                # Convert to JSON serializable format
                recommendations = self.analysis_results["recommendations"]["visualizations"]
                return self._convert_to_json_serializable(recommendations)
            return []
        except Exception as e:
            return [{"error": f"Error getting visualization recommendations: {str(e)}"}]
    
    def get_data_cleaning_recommendations(self):
        """Get specific data cleaning recommendations"""
        try:
            if not self.analysis_results or "recommendations" not in self.analysis_results:
                try:
                    self.analyze_dataset()
                except Exception as e:
                    return [{"error": f"Could not analyze dataset: {str(e)}"}]
                
            if "recommendations" in self.analysis_results and "data_cleaning" in self.analysis_results["recommendations"]:
                # Convert to JSON serializable format
                recommendations = self.analysis_results["recommendations"]["data_cleaning"]
                return self._convert_to_json_serializable(recommendations)
            return []
        except Exception as e:
            return [{"error": f"Error getting data cleaning recommendations: {str(e)}"}]