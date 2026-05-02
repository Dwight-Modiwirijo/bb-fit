FROM --platform=linux/arm64 mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Build context = repo-root; orchestrator broncode zit in orchestrator/
COPY orchestrator/BbFit.Orchestrator/BbFit.Orchestrator.csproj BbFit.Orchestrator/
RUN dotnet restore BbFit.Orchestrator/BbFit.Orchestrator.csproj

COPY orchestrator/BbFit.Orchestrator/ BbFit.Orchestrator/
RUN dotnet publish BbFit.Orchestrator/BbFit.Orchestrator.csproj \
    -c Release -o /app/publish --no-restore

FROM --platform=linux/arm64 mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

COPY --from=build /app/publish .

# ONNX model en metadata (repo-root)
COPY bbfit.onnx      ./bbfit.onnx
COPY bbfit.meta.json ./bbfit.meta.json

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "BbFit.Orchestrator.dll"]
