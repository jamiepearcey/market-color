import js from "@eslint/js";
import eslintConfigPrettier from "eslint-config-prettier";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: ["dist/**", "node_modules/**", "src-tauri/**"],
    linterOptions: {
      reportUnusedDisableDirectives: "off",
    },
  },
  {
    ...js.configs.recommended,
    files: ["server/**/*.mjs", "scripts/**/*.mjs", "*.js"],
    languageOptions: {
      ...js.configs.recommended.languageOptions,
      globals: {
        ...globals.es2022,
        ...globals.node,
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      "no-control-regex": "off",
      "no-useless-assignment": "off",
      "no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  ...tseslint.configs.recommended.map((config) => ({
    ...config,
    files: ["src/**/*.{ts,tsx}", "vite.config.ts"],
  })),
  {
    files: ["src/**/*.{ts,tsx}", "vite.config.ts"],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.es2022,
      },
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
        },
      ],
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
    plugins: {
      "react-hooks": reactHooks,
    },
  },
  eslintConfigPrettier,
];
