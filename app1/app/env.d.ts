/*
 Informa ao TypeScript que o módulo @env existe e que a 
 variável GOOGLE_API_TOKEN é do tipo string
*/
declare module "@env" {
  export const GOOGLE_API_TOKEN: string;
  export const BACKEND_URL: string;
}
